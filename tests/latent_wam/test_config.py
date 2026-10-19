from dataclasses import asdict
from pathlib import Path

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.latent_wam.config import LatentWAMConfig, TrainingConfig, TransformerConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_ROOT = REPOSITORY_ROOT / "navsim/planning/script/config/training"


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
    assert config.training.temporal_attention_mask == "block_diagonal"
    assert config.training.export_safetensors is False


def test_transformer_dimension_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        LatentWAMConfig(transformer=TransformerConfig(model_dim=255, num_heads=8))


def test_hydra_config_builds_typed_config() -> None:
    hydra_config = OmegaConf.load("navsim/planning/script/config/common/agent/latent_wam_agent.yaml")

    config = instantiate(hydra_config.config)

    assert isinstance(config, LatentWAMConfig)
    assert config.data.time_offsets_s == (-1.5, 0.0, 2.0, 4.0)
    assert config.training.temporal_attention_mask == "block_diagonal"


@pytest.mark.parametrize("mode", ["block_diagonal", "block_causal"])
def test_temporal_attention_mode_loads_from_json_and_hydra(mode: str) -> None:
    config = LatentWAMConfig.from_dict({"training": {"temporal_attention_mask": mode}})
    assert LatentWAMConfig.from_dict(asdict(config)).training.temporal_attention_mask == mode

    hydra_config = OmegaConf.load("navsim/planning/script/config/common/agent/latent_wam_agent.yaml")
    hydra_config.config.training.temporal_attention_mask = mode
    assert instantiate(hydra_config.config).training.temporal_attention_mask == mode


def test_legacy_config_keeps_diagonal_temporal_attention() -> None:
    assert LatentWAMConfig.from_dict({"training": {"learning_rate": 1e-4}}).training.temporal_attention_mask == (
        "block_diagonal"
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_safetensors_export_setting_loads_from_config_and_hydra(enabled: bool) -> None:
    config = LatentWAMConfig.from_dict({"training": {"export_safetensors": enabled}})
    assert LatentWAMConfig.from_dict(asdict(config)).training.export_safetensors is enabled
    hydra_config = OmegaConf.load("navsim/planning/script/config/common/agent/latent_wam_agent.yaml")
    hydra_config.config.training.export_safetensors = enabled
    assert instantiate(hydra_config.config).training.export_safetensors is enabled


@pytest.mark.parametrize("value", ["true", 1, None])
def test_invalid_safetensors_export_setting_is_rejected(value) -> None:
    with pytest.raises(ValueError, match="export_safetensors"):
        LatentWAMConfig(training=TrainingConfig(export_safetensors=value))


@pytest.mark.parametrize("mode", ["causal", "", None])
def test_invalid_temporal_attention_mode_is_rejected(mode: str | None) -> None:
    with pytest.raises(ValueError, match="temporal_attention_mask"):
        LatentWAMConfig(training=TrainingConfig(temporal_attention_mask=mode))


def test_all_trainval_log_split_combines_default_train_and_validation_logs() -> None:
    default_split = OmegaConf.load(TRAINING_CONFIG_ROOT / "default_train_val_test_log_split.yaml")
    all_trainval_split = OmegaConf.load(TRAINING_CONFIG_ROOT / "all_train_val_test_log_split.yaml")

    expected_train_logs = set(default_split.train_logs) | set(default_split.val_logs)
    assert set(all_trainval_split.train_logs) == expected_train_logs
    assert len(all_trainval_split.train_logs) == len(default_split.train_logs) + len(default_split.val_logs)
    assert all_trainval_split.val_logs == default_split.val_logs
    assert all_trainval_split.test_logs == default_split.test_logs
