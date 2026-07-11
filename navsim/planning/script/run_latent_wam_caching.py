"""Build versioned Latent WAM training caches with WorldMirror supervision."""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm

from navsim.agents.latent_wam.features import CAMERA_NAMES, TRAINING_FRAME_INDICES, build_camera_sensor_config
from navsim.agents.latent_wam.geometry import WorldMirrorFeatureExtractor, WorldMirrorInputs
from navsim.common.dataclasses import Scene, SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle

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
    """Adjust camera intrinsics for the legacy resize and center crop."""
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
    """Build calibrated WorldMirror inputs from the released NAVSIM frames."""
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


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Cache all scenes selected by the Hydra split configuration."""
    config = instantiate(cfg.agent.config)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    frame_indices = list(TRAINING_FRAME_INDICES)
    sensor_config = build_camera_sensor_config(frame_indices)
    scene_loader = SceneLoader(
        synthetic_sensor_path=Path(cfg.synthetic_sensor_path),
        original_sensor_path=Path(cfg.original_sensor_path),
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )
    extractor = WorldMirrorFeatureExtractor.from_pretrained(config.geometry, config.vision)
    cache_root = Path(cfg.cache_path)
    for token in tqdm(scene_loader.tokens, desc="Caching Latent WAM"):
        scene = scene_loader.get_scene_from_token(token)
        world_mirror_inputs = build_world_mirror_inputs(scene)
        features = {
            "images": world_mirror_inputs.images,
            "geometry": extractor(world_mirror_inputs),
            "ego_status": _scene_ego_status(scene),
        }
        features["schema_version"] = torch.tensor(config.data.schema_version)
        target = {
            "trajectory": torch.as_tensor(
                scene.get_future_trajectory(config.trajectory.num_poses).poses,
                dtype=torch.float32,
            )
        }
        sample_root = cache_root / scene.scene_metadata.log_name / scene.scene_metadata.initial_token
        sample_root.mkdir(parents=True, exist_ok=True)
        dump_feature_target_to_pickle(sample_root / "latent_wam_features_v2.gz", features)
        dump_feature_target_to_pickle(sample_root / "latent_wam_targets_v2.gz", target)
    logger.info("Cached %d Latent WAM samples", len(scene_loader))


if __name__ == "__main__":
    main()
