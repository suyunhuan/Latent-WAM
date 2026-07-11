from pathlib import Path

import pytest
import torch

from navsim.agents.latent_wam.checkpoint import migrate_state_dict


def test_migrate_state_dict_renames_and_drops_legacy_keys() -> None:
    state_dict = {
        "agent._transfuser_model.lora_dino_encoder.layer.weight": torch.ones(1),
        "agent._transfuser_model._trajectory_head._mlp.0.weight": torch.ones(1),
        "agent._transfuser_model.temporal_world_model.layer.weight": torch.ones(1),
        "agent._transfuser_model.target_dino_encoder.layer.weight": torch.ones(1),
        "agent._transfuser_model.refine_block.layer.weight": torch.ones(1),
    }

    migrated, dropped = migrate_state_dict(state_dict)

    assert set(migrated) == {
        "vision_encoder.layer.weight",
        "trajectory_head._mlp.0.weight",
        "temporal_world_model.layer.weight",
    }
    assert len(dropped) == 2


def test_migrate_state_dict_rejects_collisions() -> None:
    state_dict = {
        "agent._transfuser_model.lora_dino_encoder.weight": torch.ones(1),
        "vision_encoder.weight": torch.ones(1),
    }

    try:
        migrate_state_dict(state_dict)
    except ValueError as error:
        assert "collision" in str(error)
    else:
        raise AssertionError("expected migration collision")


@pytest.mark.skipif(not Path("epoch84.ckpt").is_file(), reason="private release checkpoint is unavailable")
def test_real_epoch84_checkpoint_migration_counts() -> None:
    checkpoint = torch.load("epoch84.ckpt", map_location="cpu", weights_only=True, mmap=True)

    migrated, dropped = migrate_state_dict(checkpoint["state_dict"])

    assert len(checkpoint["state_dict"]) == 1063
    assert len(migrated) == 444
    assert len(dropped) == 619
