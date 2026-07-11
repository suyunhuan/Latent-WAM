import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.latent_wam.config import LatentWAMConfig, TransformerConfig


def test_released_checkpoint_dimensions() -> None:
    config = LatentWAMConfig()

    assert len(config.data.time_offsets_s) == 4
    assert config.data.num_views == 3
    assert config.data.scene_tokens_per_view == 16
    assert config.vision.feature_dim == 768
    assert config.transformer.model_dim == 256
    assert config.trajectory.num_modes == 4
    assert config.trajectory.num_poses == 8
    assert config.training.learning_rate == pytest.approx(2e-4)
    assert config.training.target_momentum == pytest.approx(0.999)


def test_transformer_dimension_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LatentWAMConfig(transformer=TransformerConfig(model_dim=255, num_heads=8))


def test_hydra_config_builds_typed_config() -> None:
    hydra_config = OmegaConf.load("navsim/planning/script/config/common/agent/latent_wam_agent.yaml")

    config = instantiate(hydra_config.config)

    assert isinstance(config, LatentWAMConfig)
    assert config.data.time_offsets_s == (-1.5, 0.0, 2.0, 4.0)
