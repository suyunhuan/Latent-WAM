"""Feature and target builders for Latent WAM."""

from typing import Dict

import numpy as np
import torch

from navsim.agents.latent_wam.config import LatentWAMConfig
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

CAMERA_NAMES = ("cam_l0", "cam_f0", "cam_r0")
TRAINING_FRAME_INDICES = (0, 3, 7, 11)
CURRENT_FRAME_INDEX = 3


def build_camera_sensor_config(frame_indices: list[int]) -> SensorConfig:
    """Build the three-camera sensor selection used by Latent WAM.

    Args:
        frame_indices: History-frame indices to load for every camera view.

    Returns:
        A NAVSIM sensor configuration for left, front, and right cameras.
    """
    return SensorConfig(
        cam_f0=frame_indices,
        cam_l0=frame_indices,
        cam_l1=False,
        cam_l2=False,
        cam_r0=frame_indices,
        cam_r1=False,
        cam_r2=False,
        cam_b0=False,
        lidar_pc=False,
    )


def _camera_images(agent_input: AgentInput, frame_index: int) -> torch.Tensor:
    cameras = agent_input.cameras[frame_index]
    images = [getattr(cameras, camera_name).image for camera_name in CAMERA_NAMES]
    if any(image is None for image in images):
        raise ValueError(f"missing camera image at frame {frame_index}")
    return torch.from_numpy(np.stack(images, axis=0))


def _ego_status(agent_input: AgentInput, frame_index: int) -> torch.Tensor:
    status = agent_input.ego_statuses[frame_index]
    return torch.cat(
        [
            torch.as_tensor(status.driving_command, dtype=torch.float32),
            torch.as_tensor(status.ego_velocity, dtype=torch.float32),
            torch.as_tensor(status.ego_acceleration, dtype=torch.float32),
        ]
    )


class LatentWAMFeatureBuilder(AbstractFeatureBuilder):
    """Build the current-frame inputs used by public trajectory inference."""

    def __init__(self, config: LatentWAMConfig) -> None:
        self.config = config

    def get_unique_name(self) -> str:
        return "latent_wam_features_v2"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        if len(agent_input.cameras) <= CURRENT_FRAME_INDEX:
            raise ValueError("Latent WAM requires four history slots with current frame at index 3")
        return {
            "images": _camera_images(agent_input, CURRENT_FRAME_INDEX),
            "ego_status": _ego_status(agent_input, CURRENT_FRAME_INDEX),
        }


class LatentWAMTargetBuilder(AbstractTargetBuilder):
    """Build only the trajectory target consumed by Latent WAM."""

    def __init__(self, config: LatentWAMConfig) -> None:
        self.config = config

    def get_unique_name(self) -> str:
        return "latent_wam_targets_v2"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        trajectory = scene.get_future_trajectory(num_trajectory_frames=self.config.trajectory.num_poses).poses
        return {"trajectory": torch.as_tensor(trajectory, dtype=torch.float32)}


def build_training_images_and_status(agent_input: AgentInput) -> Dict[str, torch.Tensor]:
    """Build the released four-frame temporal inputs before geometry extraction."""
    required_frames = max(TRAINING_FRAME_INDICES) + 1
    if len(agent_input.cameras) < required_frames or len(agent_input.ego_statuses) < required_frames:
        raise ValueError(f"training requires at least {required_frames} frames")
    return {
        "images": torch.stack([_camera_images(agent_input, index) for index in TRAINING_FRAME_INDICES]),
        "ego_status": torch.stack([_ego_status(agent_input, index) for index in TRAINING_FRAME_INDICES]),
    }
