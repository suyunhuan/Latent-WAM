"""Build versioned Latent WAM training caches with WorldMirror supervision."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.latent_wam.cache import is_valid_public_cache, publish_public_cache
from navsim.agents.latent_wam.distributed_cache import (
    CacheFailure,
    DistributedCacheContext,
    RankCacheSummary,
    build_cache_manifest,
    canonical_tokens,
    plan_token_shard,
)
from navsim.agents.latent_wam.features import CAMERA_NAMES, TRAINING_FRAME_INDICES, build_camera_sensor_config
from navsim.agents.latent_wam.geometry import WorldMirrorFeatureExtractor, WorldMirrorInputs
from navsim.agents.latent_wam.schema import PUBLIC_FEATURE_NAME, PUBLIC_TARGET_NAME
from navsim.common.dataclasses import Scene, SceneFilter
from navsim.common.dataloader import SceneLoader

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "latent_wam_caching"
LEGACY_RESIZE_SIZE = (455, 256)
LEGACY_CROP_SIZE = (448, 224)


def adjust_camera_intrinsics(
    intrinsics: np.ndarray,
    original_size: tuple[int, int],
    resize_size: tuple[int, int] = LEGACY_RESIZE_SIZE,
    crop_size: tuple[int, int] = LEGACY_CROP_SIZE,
) -> np.ndarray:
    """Adjust camera intrinsics for the legacy resize and center crop.

    Args:
        intrinsics: Original camera matrix shaped ``[3,3]``.
        original_size: Original image size as ``(width, height)``.
        resize_size: Intermediate image size as ``(width, height)``.
        crop_size: Final center-crop size as ``(width, height)``.

    Returns:
        The adjusted float32 camera matrix shaped ``[3,3]``.
    """
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


def build_world_mirror_inputs(scene: Scene) -> WorldMirrorInputs:
    """Build calibrated WorldMirror inputs from the released NAVSIM frames.

    Args:
        scene: NAVSIM scene containing the four selected frames and cameras.

    Returns:
        Time/view-aligned images, intrinsics, and world-to-camera extrinsics.

    Raises:
        ValueError: If an image or camera calibration field is missing or invalid.
    """
    images = []
    intrinsics = []
    extrinsics = []
    for frame_index in TRAINING_FRAME_INDICES:
        frame_images = []
        frame_intrinsics = []
        frame_extrinsics = []
        for camera_name in CAMERA_NAMES:
            camera = getattr(scene.frames[frame_index].cameras, camera_name)
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
    inputs = WorldMirrorInputs(
        images=torch.stack(images),
        intrinsics=torch.stack(intrinsics),
        extrinsics=torch.stack(extrinsics),
    )
    inputs.validate()
    return inputs


def _scene_ego_status(scene: Scene) -> torch.Tensor:
    """Collect the four ego-status vectors used by Latent WAM training."""
    statuses = []
    for frame_index in TRAINING_FRAME_INDICES:
        frame = scene.frames[frame_index]
        statuses.append(
            torch.cat(
                [
                    torch.as_tensor(frame.ego_status.driving_command, dtype=torch.float32),
                    torch.as_tensor(frame.ego_status.ego_velocity, dtype=torch.float32),
                    torch.as_tensor(frame.ego_status.ego_acceleration, dtype=torch.float32),
                ]
            )
        )
    return torch.stack(statuses)


def _token_to_log_name(scene_loader: SceneLoader) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for log_name, tokens in scene_loader.get_tokens_list_per_log().items():
        for token in tokens:
            if token in mapping:
                raise ValueError(f"cache token belongs to multiple logs: {token}")
            mapping[token] = log_name
    return mapping


def _cache_paths(cache_root: Path, log_name: str, token: str) -> Tuple[Path, Path]:
    sample_root = cache_root / log_name / token
    return sample_root / PUBLIC_FEATURE_NAME, sample_root / PUBLIC_TARGET_NAME


def _write_manifest(cache_root: Path, manifest: dict) -> Path:
    metadata_root = cache_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)
    manifest_path = metadata_root / "cache_manifest.json"
    temporary_path = metadata_root / f"cache_manifest.rank_0.pid_{os.getpid()}.tmp"
    try:
        temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        json.loads(temporary_path.read_text(encoding="utf-8"))
        os.replace(temporary_path, manifest_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return manifest_path


def _run_cache(cfg: DictConfig, context: DistributedCacheContext) -> None:
    config = instantiate(cfg.agent.config)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    sensor_config = build_camera_sensor_config(list(TRAINING_FRAME_INDICES))
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    all_tokens = canonical_tokens(scene_loader.tokens)
    context.verify_token_set(all_tokens)
    assigned_tokens = plan_token_shard(all_tokens, context.world_size, context.rank)
    token_to_log = _token_to_log_name(scene_loader)
    missing_log_names = set(all_tokens).difference(token_to_log)
    if missing_log_names:
        raise ValueError(f"cache tokens have no log mapping: {sorted(missing_log_names)}")
    cache_root = Path(cfg.cache_path)
    logger.info(
        "Rank %d/%d assigned %d of %d cache tokens",
        context.rank,
        context.world_size,
        len(assigned_tokens),
        len(all_tokens),
    )

    if cfg.distributed_dry_run:
        summary = RankCacheSummary(rank=context.rank, assigned=len(assigned_tokens))
    else:
        skipped_tokens: List[str] = []
        pending_tokens: List[str] = []
        for token in assigned_tokens:
            feature_path, target_path = _cache_paths(cache_root, token_to_log[token], token)
            if not cfg.force_cache_computation and is_valid_public_cache(
                feature_path,
                target_path,
                expected_num_poses=config.trajectory.num_poses,
            ):
                skipped_tokens.append(token)
            else:
                pending_tokens.append(token)

        extractor = None
        if pending_tokens:
            extractor = WorldMirrorFeatureExtractor.from_pretrained(config.geometry, config.vision)
        succeeded_tokens: List[str] = []
        failures: List[CacheFailure] = []
        for token in tqdm(pending_tokens, desc=f"Caching Latent WAM rank {context.rank}"):
            log_name = token_to_log[token]
            try:
                assert extractor is not None
                scene = scene_loader.get_scene_from_token(token)
                world_mirror_inputs = build_world_mirror_inputs(scene)
                features = {
                    "images": world_mirror_inputs.images,
                    "geometry": extractor(world_mirror_inputs),
                    "ego_status": _scene_ego_status(scene),
                    "schema_version": torch.tensor(config.data.schema_version),
                }
                targets = {
                    "trajectory": torch.as_tensor(
                        scene.get_future_trajectory(config.trajectory.num_poses).poses,
                        dtype=torch.float32,
                    )
                }
                sample_root = cache_root / log_name / token
                publish_public_cache(
                    sample_root,
                    features,
                    targets,
                    rank=context.rank,
                    expected_num_poses=config.trajectory.num_poses,
                )
                succeeded_tokens.append(token)
            except Exception as error:
                logger.exception("Rank %d failed to cache token %s", context.rank, token)
                failures.append(
                    CacheFailure(
                        token=token,
                        log_name=log_name,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
        summary = RankCacheSummary(
            rank=context.rank,
            assigned=len(assigned_tokens),
            succeeded=len(succeeded_tokens),
            skipped=len(skipped_tokens),
            failures=tuple(failures),
        )

    summaries = context.all_gather_object(summary)
    manifest = build_cache_manifest(
        summaries=summaries,
        total_tokens=len(all_tokens),
        world_size=context.world_size,
        dry_run=bool(cfg.distributed_dry_run),
    )
    if context.rank == 0:
        manifest_path = _write_manifest(cache_root, manifest)
        logger.info("Cache manifest written to %s", manifest_path)
    context.barrier()
    if manifest["failed"]:
        raise RuntimeError(f"cache generation failed for {manifest['failed']} tokens")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Cache all scenes selected by the Hydra split configuration."""
    context = DistributedCacheContext.from_environment()
    with context.activate():
        _run_cache(cfg, context)


if __name__ == "__main__":
    main()
