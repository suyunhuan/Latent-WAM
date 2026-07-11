"""Build the frame-deduplicated sharded Latent WAM training cache."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.latent_wam.cache import (
    MANIFEST_NAME,
    FrameRecord,
    SampleRecord,
    fingerprint_digest,
    load_cache_manifest,
    load_cache_shard_record,
    publish_cache_index,
    read_partial_cache_indices,
    sha256_file,
    validate_cache_headers,
    write_cache_shard,
    write_partial_cache_index,
)
from navsim.agents.latent_wam.distributed_cache import (
    DistributedCacheContext,
    FrameCachePlan,
    PlannedFrame,
    build_frame_cache_plan,
    plan_shard_ids,
)
from navsim.agents.latent_wam.features import CAMERA_NAMES, CURRENT_FRAME_INDEX, TRAINING_FRAME_INDICES
from navsim.agents.latent_wam.geometry import WorldMirrorFeatureExtractor, WorldMirrorInputs
from navsim.agents.latent_wam.preprocessing import (
    DINO_CROP_SIZE,
    DINO_RESIZE_SIZE,
    PREPROCESSING_REVISION,
    preprocess_camera_images,
)
from navsim.common.dataclasses import Cameras, Scene, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "latent_wam_caching"


def adjust_camera_intrinsics(
    intrinsics: np.ndarray,
    original_size: Tuple[int, int],
    resize_size: Tuple[int, int] = (DINO_RESIZE_SIZE[1], DINO_RESIZE_SIZE[0]),
    crop_size: Tuple[int, int] = (DINO_CROP_SIZE[1], DINO_CROP_SIZE[0]),
) -> np.ndarray:
    """Adjust a camera matrix for deterministic resize and center crop."""
    original_width, original_height = original_size
    resize_width, resize_height = resize_size
    crop_width, crop_height = crop_size
    scale_x = resize_width / original_width
    scale_y = resize_height / original_height
    crop_offset_x = (resize_width - crop_width) / 2
    crop_offset_y = (resize_height - crop_height) / 2
    adjusted = np.asarray(intrinsics, dtype=np.float32).copy()
    adjusted[0, 0] *= scale_x
    adjusted[1, 1] *= scale_y
    adjusted[0, 2] = adjusted[0, 2] * scale_x - crop_offset_x
    adjusted[1, 2] = adjusted[1, 2] * scale_y - crop_offset_y
    return adjusted


def _world_mirror_inputs_from_cameras(
    cameras_per_frame: Sequence[Cameras], frame_labels: Sequence[int] | None = None
) -> WorldMirrorInputs:
    images = []
    intrinsics = []
    extrinsics = []
    labels = list(range(len(cameras_per_frame))) if frame_labels is None else list(frame_labels)
    if len(labels) != len(cameras_per_frame):
        raise ValueError("frame labels must align with the camera batch")
    for frame_index, cameras in zip(labels, cameras_per_frame):
        frame_images = []
        frame_intrinsics = []
        frame_extrinsics = []
        for camera_name in CAMERA_NAMES:
            camera = getattr(cameras, camera_name)
            for field_name in ("image", "intrinsics", "sensor2lidar_rotation", "sensor2lidar_translation"):
                if getattr(camera, field_name) is None:
                    raise ValueError(f"frame {frame_index} camera {camera_name} is missing {field_name}")
            image = np.asarray(camera.image)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"frame {frame_index} camera {camera_name} image must have shape [H,W,3]")
            image_height, image_width = image.shape[:2]
            frame_images.append(torch.from_numpy(image))
            frame_intrinsics.append(
                torch.from_numpy(adjust_camera_intrinsics(camera.intrinsics, (image_width, image_height)))
            )
            camera_to_world = np.eye(4, dtype=np.float32)
            camera_to_world[:3, :3] = camera.sensor2lidar_rotation
            camera_to_world[:3, 3] = camera.sensor2lidar_translation
            frame_extrinsics.append(torch.from_numpy(np.linalg.inv(camera_to_world).astype(np.float32)))
        images.append(torch.stack(frame_images))
        intrinsics.append(torch.stack(frame_intrinsics))
        extrinsics.append(torch.stack(frame_extrinsics))
    result = WorldMirrorInputs(
        images=torch.stack(images),
        intrinsics=torch.stack(intrinsics),
        extrinsics=torch.stack(extrinsics),
    )
    result.validate()
    return result


def build_world_mirror_inputs(scene: Scene) -> WorldMirrorInputs:
    """Build WorldMirror inputs for the four training frames of an existing scene."""
    return _world_mirror_inputs_from_cameras(
        [scene.frames[index].cameras for index in TRAINING_FRAME_INDICES], frame_labels=TRAINING_FRAME_INDICES
    )


def _load_frame_cameras(frame: Mapping[str, Any], sensor_root: Path) -> Cameras:
    return Cameras.from_camera_dict(
        sensor_blobs_path=sensor_root,
        camera_dict=frame["cams"],
        sensor_names=list(CAMERA_NAMES),
    )


def _ego_status(frame: Mapping[str, Any]) -> torch.Tensor:
    dynamic = torch.as_tensor(frame["ego_dynamic_state"], dtype=torch.float32)
    command = torch.as_tensor(frame["driving_command"], dtype=torch.float32)
    status = torch.cat((command, dynamic))
    if status.shape != (8,):
        raise ValueError("raw NAVSIM ego status must contain four commands and four dynamic values")
    return status


def _sample_trajectory(scene_frames: Sequence[Mapping[str, Any]], num_poses: int) -> torch.Tensor:
    trajectory_frames = scene_frames[CURRENT_FRAME_INDEX : CURRENT_FRAME_INDEX + num_poses + 1]
    if len(trajectory_frames) != num_poses + 1:
        raise ValueError("sample does not contain enough future frames for its trajectory")
    poses = np.stack([Scene._build_ego_status(frame).ego_pose for frame in trajectory_frames])
    relative = convert_absolute_to_relative_se2_array(StateSE2(*poses[0]), poses[1:])
    return torch.as_tensor(relative, dtype=torch.float32)


def _frames_in_shard(plan: FrameCachePlan, shard_id: int) -> List[PlannedFrame]:
    return [frame for frame in plan.frames if frame.shard_id == shard_id]


def _generate_shard(
    cache_root: Path,
    shard_id: int,
    frames: Sequence[PlannedFrame],
    plan: FrameCachePlan,
    scenes: Mapping[str, Sequence[Mapping[str, Any]]],
    sensor_root: Path,
    extractor: WorldMirrorFeatureExtractor,
    extraction_batch_frames: int,
    cache_fingerprint_sha256: str,
) -> Dict[str, Any]:
    frame_count = len(frames)
    image_buffer = torch.empty((frame_count, 3, *DINO_CROP_SIZE, 3), dtype=torch.uint8)
    geometry_buffer: torch.Tensor | None = None
    ego_status_buffer = torch.empty((frame_count, 8), dtype=torch.float32)
    for begin in range(0, frame_count, extraction_batch_frames):
        batch_frames = frames[begin : begin + extraction_batch_frames]
        raw_frames = [
            scenes[plan.sources[frame.token].sample_token][plan.sources[frame.token].frame_index]
            for frame in batch_frames
        ]
        cameras = [_load_frame_cameras(frame, sensor_root) for frame in raw_frames]
        inputs = _world_mirror_inputs_from_cameras(cameras)
        processed_images = preprocess_camera_images(inputs.images)
        geometry = extractor(inputs)
        if processed_images.shape[1:] != (3, *DINO_CROP_SIZE, 3):
            raise ValueError(f"preprocessed image shape is incompatible: {tuple(processed_images.shape)}")
        if geometry.shape[1:] != (3, 519, 2048) or geometry.dtype != torch.float32:
            raise ValueError(f"WorldMirror geometry shape or dtype is incompatible: {geometry.shape}/{geometry.dtype}")
        if geometry_buffer is None:
            geometry_buffer = torch.empty((frame_count, *geometry.shape[1:]), dtype=torch.float32)
        end = begin + len(batch_frames)
        image_buffer[begin:end].copy_(processed_images)
        geometry_buffer[begin:end].copy_(geometry)
        ego_status_buffer[begin:end].copy_(torch.stack([_ego_status(frame) for frame in raw_frames]))
    assert geometry_buffer is not None
    return write_cache_shard(
        cache_root,
        shard_id,
        [frame.token for frame in frames],
        image_buffer,
        geometry_buffer,
        ego_status_buffer,
        cache_fingerprint_sha256,
    )


def _cache_fingerprint(cfg: DictConfig, plan: FrameCachePlan) -> Dict[str, Any]:
    worldmirror_weights = Path(str(cfg.agent.config.geometry.model_id)).expanduser() / "model.safetensors"
    if not worldmirror_weights.is_file():
        raise ValueError(f"WorldMirror model.safetensors is required for cache identity: {worldmirror_weights}")
    return {
        "dataset": {"name": "NAVSIM", "version": "2.1.2", "split": str(cfg.train_test_split.data_split)},
        "plan_sha256": plan.fingerprint(),
        "sample_count": len(plan.samples),
        "logs": sorted({sample.log_name for sample in plan.samples}),
        "time_offsets_s": list(cfg.agent.config.data.time_offsets_s),
        "camera_names": list(CAMERA_NAMES),
        "preprocessing": {
            "revision": PREPROCESSING_REVISION,
            "resize_size": list(DINO_RESIZE_SIZE),
            "crop_size": list(DINO_CROP_SIZE),
            "interpolation": "torch_bicubic_antialias",
        },
        "worldmirror": {
            "model_id": str(cfg.agent.config.geometry.model_id),
            "weights_sha256": sha256_file(worldmirror_weights),
            "patch_start_index": int(cfg.agent.config.geometry.patch_start_index),
        },
        "geometry": {"shape_per_frame": [3, 519, 2048], "dtype": "float32"},
    }


def _frame_records(frames: Iterable[PlannedFrame]) -> List[FrameRecord]:
    return [
        FrameRecord(frame.token, frame.log_name, frame.timestamp, frame.shard_id, frame.row_index) for frame in frames
    ]


def _sample_records(
    samples: Iterable[Any], scenes: Mapping[str, Sequence[Mapping[str, Any]]], num_poses: int
) -> List[SampleRecord]:
    return [
        SampleRecord(
            token=sample.token,
            log_name=sample.log_name,
            frame_tokens=sample.frame_tokens,
            trajectory=_sample_trajectory(scenes[sample.token], num_poses),
        )
        for sample in samples
    ]


def _run_cache(cfg: DictConfig, context: DistributedCacheContext) -> None:
    config = instantiate(cfg.agent.config)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=None,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    if scene_loader.synthetic_scenes:
        raise ValueError("Latent WAM cache generation supports released original NAVSIM scenes only")
    plan = build_frame_cache_plan(
        scene_loader.scene_frames_dicts,
        frame_indices=TRAINING_FRAME_INDICES,
        frames_per_shard=int(cfg.frames_per_shard),
    )
    context.verify_plan(plan)
    fingerprint = context.broadcast_object(_cache_fingerprint(cfg, plan) if context.rank == 0 else None)
    cache_fingerprint_sha256 = fingerprint_digest(fingerprint)
    cache_root = Path(cfg.cache_path)
    manifest_path = cache_root / MANIFEST_NAME
    cache_state = None
    cache_error = None
    if context.rank == 0 and manifest_path.is_file() and not cfg.force_cache_computation:
        try:
            manifest = load_cache_manifest(
                cache_root,
                expected_fingerprint=fingerprint,
                verify_presence=False,
            )
        except ValueError as error:
            cache_error = str(error)
        else:
            try:
                manifest = load_cache_manifest(
                    cache_root,
                    expected_fingerprint=fingerprint,
                    verify_files=bool(cfg.verify_resume_checksums),
                )
                validate_cache_headers(cache_root, manifest)
                cache_state = "complete"
            except ValueError:
                cache_state = "repair"
    cache_state, cache_error = context.broadcast_object((cache_state, cache_error))
    if cache_error:
        raise ValueError(cache_error)
    if cache_state == "complete":
        logger.info("Complete compatible cache already exists at %s", cache_root)
        return
    if cfg.distributed_dry_run:
        logger.info(
            "Verified cache plan: %d samples, %d unique frames, %d shards across %d ranks",
            len(plan.samples),
            len(plan.frames),
            plan.num_shards,
            context.world_size,
        )
        return
    if context.rank == 0 and manifest_path.exists():
        manifest_path.unlink()
    context.barrier()

    owned_shards = plan_shard_ids(plan.num_shards, context.world_size, context.rank)
    shard_records = []
    pending = []
    for shard_id in owned_shards:
        frames = _frames_in_shard(plan, shard_id)
        existing = None
        if not cfg.force_cache_computation:
            existing = load_cache_shard_record(
                cache_root,
                shard_id,
                [frame.token for frame in frames],
                cache_fingerprint_sha256,
                verify_files=bool(cfg.verify_resume_checksums),
            )
        if existing is None:
            pending.append((shard_id, frames))
        else:
            shard_records.append(existing)

    extractor = None
    failures = []
    if pending:
        try:
            extractor = WorldMirrorFeatureExtractor.from_pretrained(
                config.geometry,
                config.vision,
                device=torch.device("cuda", context.local_rank) if context.cuda_available else torch.device("cpu"),
            )
        except Exception as error:
            failures.append(f"rank {context.rank} failed to initialize WorldMirror: {type(error).__name__}: {error}")
    if extractor is not None:
        for shard_id, frames in tqdm(pending, desc=f"Caching Latent WAM rank {context.rank}"):
            try:
                shard_records.append(
                    _generate_shard(
                        cache_root,
                        shard_id,
                        frames,
                        plan,
                        scene_loader.scene_frames_dicts,
                        Path(cfg.original_sensor_path),
                        extractor,
                        int(cfg.extraction_batch_frames),
                        cache_fingerprint_sha256,
                    )
                )
            except Exception as error:
                logger.exception("Rank %d failed to build shard %d", context.rank, shard_id)
                failures.append(f"rank {context.rank} shard {shard_id}: {type(error).__name__}: {error}")

    all_failures = context.all_gather_object(failures)
    if any(all_failures):
        raise RuntimeError(f"cache generation failed: {all_failures}")

    owned_frames = [frame for frame in plan.frames if frame.shard_id in set(owned_shards)]
    owned_samples = [sample for index, sample in enumerate(plan.samples) if index % context.world_size == context.rank]
    write_partial_cache_index(
        cache_root,
        context.rank,
        _frame_records(owned_frames),
        _sample_records(owned_samples, scene_loader.scene_frames_dicts, config.trajectory.num_poses),
    )
    gathered_shards = context.all_gather_object(shard_records)
    context.barrier()

    publish_error = None
    if context.rank == 0:
        try:
            frames, samples = read_partial_cache_indices(cache_root, context.world_size)
            all_shards = [shard for rank_shards in gathered_shards for shard in rank_shards]
            publish_cache_index(
                cache_root,
                frames=frames,
                samples=samples,
                shards=all_shards,
                fingerprint=fingerprint,
                frames_per_shard=int(cfg.frames_per_shard),
            )
            logger.info("Published complete Latent WAM cache at %s", cache_root)
        except Exception as error:
            logger.exception("Failed to publish final cache manifest")
            publish_error = f"{type(error).__name__}: {error}"
    publish_error = context.broadcast_object(publish_error)
    if publish_error:
        raise RuntimeError(f"cache index publication failed: {publish_error}")
    context.barrier()


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate all shards selected by the Hydra scene split."""
    context = DistributedCacheContext.from_environment()
    with context.activate():
        _run_cache(cfg, context)


if __name__ == "__main__":
    main()
