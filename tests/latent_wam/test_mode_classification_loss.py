import torch

from navsim.agents.latent_wam.config import LatentWAMConfig, LossConfig
from navsim.agents.latent_wam.model import LatentWAMModel


def _loss_model(**loss_kwargs) -> LatentWAMModel:
    """Bare model instance that only needs ``config`` for ``compute_loss``."""
    model = LatentWAMModel.__new__(LatentWAMModel)
    torch.nn.Module.__init__(model)
    model.config = LatentWAMConfig(losses=LossConfig(**loss_kwargs))
    return model


def _aux_predictions(batch: int = 2, modes: int = 4, poses: int = 8) -> dict:
    return {
        "wm_next_latent": torch.zeros(batch, 3, 4, 8),
        "gt_next_latent": torch.zeros(batch, 3, 4, 8),
        "geometry_feature_align": torch.randn(batch, 4, 3, 8, 16),
        "geometry_feature_gt": torch.randn(batch, 4, 3, 8, 16),
        "cmd_pred": torch.randn(batch, 3, 4),
        "cmd_gt": torch.zeros(batch, 3, dtype=torch.long),
        "velocity_pred": torch.zeros(batch, 3, 2),
        "velocity_gt": torch.zeros(batch, 3, 2),
        "acceleration_pred": torch.zeros(batch, 3, 2),
        "acceleration_gt": torch.zeros(batch, 3, 2),
        "mode_command_gt": torch.zeros(batch, dtype=torch.long),
    }


def test_default_loss_dict_omits_mode_ce() -> None:
    model = _loss_model()
    ground_truth = torch.zeros(2, 8, 3)
    all_traj = torch.randn(2, 4, 8, 3)
    all_traj[:, 2, -1, :2] = 0.0
    predictions = {
        "all_trajectories": all_traj,
        "mode_logits": torch.randn(2, 4, requires_grad=True),
        **_aux_predictions(),
    }
    losses = model.compute_loss({"trajectory": ground_truth}, predictions)
    assert "mode_classification_loss" not in losses
    assert {"trajectory_loss", "wm_loss", "geometry_loss", "ego_status_loss"} <= set(losses)


def test_mode_ce_best_mode_backprops_into_logits() -> None:
    model = _loss_model(mode_classification=0.1, mode_ce_target="best_mode")
    ground_truth = torch.zeros(2, 8, 3)
    all_traj = torch.randn(2, 4, 8, 3)
    all_traj[:, 1, -1, :2] = 0.0
    logits = torch.zeros(2, 4, requires_grad=True)
    predictions = {"all_trajectories": all_traj, "mode_logits": logits, **_aux_predictions()}
    losses = model.compute_loss({"trajectory": ground_truth}, predictions)
    assert "mode_classification_loss" in losses
    assert torch.isfinite(losses["mode_classification_loss"])
    losses["mode_classification_loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_mode_ce_command_target() -> None:
    model = _loss_model(mode_classification=0.2, mode_ce_target="command")
    ground_truth = torch.zeros(2, 8, 3)
    all_traj = torch.zeros(2, 4, 8, 3)
    logits = torch.randn(2, 4, requires_grad=True)
    aux = _aux_predictions()
    aux["mode_command_gt"] = torch.tensor([1, 3])
    predictions = {"all_trajectories": all_traj, "mode_logits": logits, **aux}
    losses = model.compute_loss({"trajectory": ground_truth}, predictions)
    assert torch.isfinite(losses["mode_classification_loss"])


def test_lateral_weight_increases_pure_y_error() -> None:
    ground_truth = torch.zeros(1, 8, 3)
    all_traj = torch.zeros(1, 2, 8, 3)
    # Only mode 0 is selectable as best (mode 1 far at endpoint); mode 0 has pure +y error.
    all_traj[0, 0, :, 1] = 1.0
    all_traj[0, 1, -1, 0] = 100.0
    base = {"all_trajectories": all_traj, "mode_logits": torch.zeros(1, 2), **_aux_predictions(batch=1, modes=2)}
    loss_iso = _loss_model(trajectory_lateral_weight=1.0).compute_loss({"trajectory": ground_truth}, base)[
        "trajectory_loss"
    ]
    loss_lat = _loss_model(trajectory_lateral_weight=2.0).compute_loss({"trajectory": ground_truth}, base)[
        "trajectory_loss"
    ]
    assert float(loss_lat) > float(loss_iso)
