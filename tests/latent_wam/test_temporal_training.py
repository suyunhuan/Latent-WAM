"""CPU-only synthetic integration checks with a small, locally initialized DINO."""

import pytest
import torch
from transformers import Dinov2Config, Dinov2Model

from navsim.agents.latent_wam.config import (
    GeometryConfig,
    LatentWAMConfig,
    TrainingConfig,
    TransformerConfig,
    VisionConfig,
)
from navsim.agents.latent_wam.model import LatentWAMModel


def _model(mode: str) -> LatentWAMModel:
    torch.manual_seed(31)
    vision = Dinov2Model(
        Dinov2Config(
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            image_size=224,
            patch_size=14,
        )
    )
    config = LatentWAMConfig(
        vision=VisionConfig(feature_dim=32),
        geometry=GeometryConfig(feature_dim=8),
        transformer=TransformerConfig(model_dim=32, feedforward_dim=64, num_heads=1, num_layers=2),
        training=TrainingConfig(temporal_attention_mask=mode),
    )
    return LatentWAMModel(config, vision_encoder=vision, build_target_encoder=True)


def _features() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(37)
    ego_status = torch.randn(1, 4, 8, generator=generator)
    ego_status[..., :4] = torch.eye(4)
    return {
        "images": torch.randint(0, 256, (1, 4, 3, 224, 448, 3), dtype=torch.uint8, generator=generator),
        "geometry": torch.randn(1, 4, 3, 519, 8, generator=generator),
        "ego_status": ego_status,
    }


@pytest.fixture(autouse=True)
def _forbid_pretrained_downloads(monkeypatch) -> None:
    def fail(*args, **kwargs):
        pytest.fail("synthetic integration tests must not download pretrained weights")

    monkeypatch.setattr("navsim.agents.latent_wam.model.AutoModel.from_pretrained", fail)


@pytest.mark.parametrize("mode", ["block_diagonal", "block_causal"])
def test_teacher_forcing_has_finite_losses_gradients_and_ema(mode: str) -> None:
    model = _model(mode).train()
    features = _features()
    predictions = model.forward_train(features)
    losses = model.compute_loss({"trajectory": torch.zeros(1, 8, 3)}, predictions)
    assert predictions["wm_next_latent"].shape == (1, 3, 49, 32)
    assert predictions["gt_next_latent"].shape == (1, 3, 49, 32)
    assert all(torch.isfinite(value) for value in losses.values())
    sum(losses.values()).backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.vision_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.temporal_world_model.parameters())
    assert all(p.grad is None for p in model.target_dino_encoder.parameters())

    target_before = {name: value.clone() for name, value in model.target_dino_encoder.named_parameters()}
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    optimizer.step()
    model.update_target_encoder(momentum=0.5)
    assert any(
        not torch.equal(target_before[name], value) for name, value in model.target_dino_encoder.named_parameters()
    )


def test_mask_mode_does_not_change_current_frame_inference_or_trajectory_branch() -> None:
    diagonal, causal = (_model(mode).eval() for mode in ("block_diagonal", "block_causal"))
    causal.load_state_dict(diagonal.state_dict(), strict=True)
    features = _features()
    with torch.no_grad():
        old = diagonal.predict(features["images"][:, 1], features["ego_status"][:, 1])
        new = causal.predict(features["images"][:, 1], features["ego_status"][:, 1])
        old_training = diagonal.forward_train(features)
        new_training = causal.forward_train(features)
    for name in ("trajectory", "all_trajectories", "mode_logits"):
        torch.testing.assert_close(new[name], old[name], rtol=0, atol=0)
        torch.testing.assert_close(new_training[name], old_training[name], rtol=0, atol=0)
    assert not torch.allclose(new_training["wm_next_latent"], old_training["wm_next_latent"])
