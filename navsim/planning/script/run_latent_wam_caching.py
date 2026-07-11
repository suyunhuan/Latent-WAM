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
from navsim.agents.latent_wam.geometry import WorldMirrorFeatureExtractor
from navsim.common.dataclasses import Scene, SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import dump_feature_target_to_pickle

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/training"
CONFIG_NAME = "latent_wam_caching"


def _scene_features(scene: Scene) -> dict[str, torch.Tensor]:
    images = []
    statuses = []
    for frame_index in TRAINING_FRAME_INDICES:
        frame = scene.frames[frame_index]
        camera_images = [getattr(frame.cameras, name).image for name in CAMERA_NAMES]
        if any(image is None for image in camera_images):
            raise ValueError(f"missing camera image at frame {frame_index}")
        images.append(torch.from_numpy(np.stack(camera_images)))
        statuses.append(
            torch.cat(
                [
                    torch.as_tensor(frame.ego_status.driving_command, dtype=torch.float32),
                    torch.as_tensor(frame.ego_status.ego_velocity, dtype=torch.float32),
                    torch.as_tensor(frame.ego_status.ego_acceleration, dtype=torch.float32),
                ]
            )
        )
    return {"images": torch.stack(images), "ego_status": torch.stack(statuses)}


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
        features = _scene_features(scene)
        features["geometry"] = extractor(features["images"])
        features["schema_version"] = torch.tensor(config.data.schema_version)
        target = {
            "trajectory": torch.as_tensor(
                scene.get_future_trajectory(config.trajectory.num_poses).poses,
                dtype=torch.float32,
            )
        }
        sample_root = cache_root / scene.scene_metadata.log_name / scene.scene_metadata.initial_token
        sample_root.mkdir(parents=True, exist_ok=True)
        dump_feature_target_to_pickle(sample_root / "latent_wam_features_v1.gz", features)
        dump_feature_target_to_pickle(sample_root / "latent_wam_targets_v1.gz", target)
    logger.info("Cached %d Latent WAM samples", len(scene_loader))


if __name__ == "__main__":
    main()
