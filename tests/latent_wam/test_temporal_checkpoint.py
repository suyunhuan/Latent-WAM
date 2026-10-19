"""Checkpoint behavior metadata is distinct from model parameter compatibility."""

import logging
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, TensorDataset

from navsim.agents.latent_wam.config import LatentWAMConfig, TrainingConfig
from navsim.agents.latent_wam.lightning_module import TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY, LatentWAMLightningModule


class _Agent(torch.nn.Module):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.build_target_encoder = True
        self.config = LatentWAMConfig(training=TrainingConfig(temporal_attention_mask=mode))
        self.weight = torch.nn.Parameter(torch.ones(1))


class _CheckpointLifecycleModule(LatentWAMLightningModule):
    """Exercise real Lightning checkpoint hooks with a one-parameter CPU model."""

    def training_step(self, batch, batch_idx):
        return self.agent.weight.square().sum()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass


@pytest.mark.parametrize("mode", ["block_diagonal", "block_causal"])
def test_checkpoint_mode_roundtrips_without_changing_state_dict(mode: str, tmp_path: Path, caplog) -> None:
    with caplog.at_level(logging.INFO):
        module = LatentWAMLightningModule(_Agent(mode))
    assert f"Latent WAM temporal attention mask: {mode}" in caplog.text
    original_keys = set(module.state_dict())
    checkpoint = {"state_dict": module.state_dict(), "optimizer_states": [{"sentinel": 7}], "epoch": 3}
    module.on_save_checkpoint(checkpoint)
    path = tmp_path / "checkpoint.ckpt"
    torch.save(checkpoint, path)
    restored = torch.load(path, weights_only=True)
    replacement = LatentWAMLightningModule(_Agent(mode))
    replacement.on_load_checkpoint(restored)
    replacement.load_state_dict(restored["state_dict"], strict=True)
    assert restored[TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY] == mode
    assert restored["optimizer_states"] == [{"sentinel": 7}]
    assert restored["epoch"] == 3
    assert original_keys == set(replacement.state_dict()) == {"agent.weight"}


def test_legacy_checkpoint_is_accepted_only_in_diagonal_mode() -> None:
    LatentWAMLightningModule(_Agent("block_diagonal")).on_load_checkpoint({})
    with pytest.raises(ValueError, match="weights-only initialization"):
        LatentWAMLightningModule(_Agent("block_causal")).on_load_checkpoint({})


@pytest.mark.parametrize("stored,requested", [("block_diagonal", "block_causal"), ("block_causal", "block_diagonal")])
def test_resume_rejects_cross_mode_without_modifying_checkpoint(stored: str, requested: str) -> None:
    checkpoint = {TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY: stored, "optimizer_states": [{"sentinel": 7}]}
    with pytest.raises(ValueError, match="cannot resume Latent WAM"):
        LatentWAMLightningModule(_Agent(requested)).on_load_checkpoint(checkpoint)
    assert checkpoint == {TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY: stored, "optimizer_states": [{"sentinel": 7}]}


def test_checkpoint_with_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot resume Latent WAM"):
        LatentWAMLightningModule(_Agent("block_diagonal")).on_load_checkpoint(
            {TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY: "unknown"}
        )


def test_lightning_fit_saves_mode_and_rejects_cross_mode_resume(tmp_path: Path) -> None:
    loader = DataLoader(TensorDataset(torch.zeros(2, 1)), batch_size=1)
    trainer_options = dict(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    module = _CheckpointLifecycleModule(_Agent("block_diagonal"))
    trainer = pl.Trainer(max_steps=1, **trainer_options)
    trainer.fit(module, train_dataloaders=loader)
    checkpoint_path = tmp_path / "one_step.ckpt"
    trainer.save_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert checkpoint[TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY] == "block_diagonal"
    assert checkpoint["global_step"] == 1
    assert checkpoint["optimizer_states"][0]["state"]

    resumed = _CheckpointLifecycleModule(_Agent("block_diagonal"))
    resume_trainer = pl.Trainer(max_steps=2, **trainer_options)
    resume_trainer.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint_path)
    assert resume_trainer.global_step == 2
    torch.testing.assert_close(resumed.agent.weight, torch.tensor([0.46]))

    mismatched = _CheckpointLifecycleModule(_Agent("block_causal"))
    mismatch_trainer = pl.Trainer(max_steps=2, **trainer_options)
    with pytest.raises(ValueError, match="cannot resume Latent WAM"):
        mismatch_trainer.fit(mismatched, train_dataloaders=loader, ckpt_path=checkpoint_path)
    assert mismatch_trainer.global_step == 0
