"""Typed Latent WAM model outputs."""

from typing import Dict, TypedDict

import torch


class TrajectoryPrediction(TypedDict):
    """Public multi-modal trajectory prediction."""

    trajectory: torch.Tensor
    all_trajectories: torch.Tensor
    mode_logits: torch.Tensor


class TrainingOutput(TrajectoryPrediction, total=False):
    """Trajectory prediction plus training-only auxiliary outputs."""

    geometry_feature_align: torch.Tensor
    geometry_feature_gt: torch.Tensor
    wm_next_latent: torch.Tensor
    gt_next_latent: torch.Tensor
    cmd_pred: torch.Tensor
    velocity_pred: torch.Tensor
    acceleration_pred: torch.Tensor
    cmd_gt: torch.Tensor
    velocity_gt: torch.Tensor
    acceleration_gt: torch.Tensor
    # Current-frame driving-command index used by the trajectory head (for mode CE).
    mode_command_gt: torch.Tensor


LossDict = Dict[str, torch.Tensor]
