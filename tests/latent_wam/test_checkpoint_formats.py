"""Serialization, paired export, and exact-resume integration regressions."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pytorch_lightning as pl
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, TensorDataset

from navsim.agents.latent_wam.checkpoint import (
    TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY,
    LatentWAMModelCheckpoint,
    load_model_weights,
    save_model_safetensors,
)
from navsim.agents.latent_wam.config import LatentWAMConfig
from navsim.agents.latent_wam.lightning_module import LatentWAMLightningModule


@pytest.mark.parametrize("prefix", ["", "agent.model."])
def test_checkpoint_loader_extracts_only_weights(tmp_path, prefix):
    path = tmp_path / "old.ckpt"
    torch.save({"state_dict": {prefix + "weight": torch.ones(2)}, "optimizer_states": [{"step": 99}]}, path)
    weights = load_model_weights(path)
    assert set(weights) == {"weight"}
    assert torch.equal(weights["weight"], torch.ones(2))


@pytest.mark.parametrize(
    "payload,message",
    [
        ({}, "state_dict mapping"),
        ({"state_dict": {}}, "non-empty mapping"),
        ({"state_dict": [1]}, "non-empty mapping"),
        ({"state_dict": {"weight": "invalid"}}, "tensor values"),
        ({"state_dict": {1: torch.ones(1)}}, "string keys"),
        ({"state_dict": {"agent.model.weight": torch.ones(1), "other": torch.ones(1)}}, "mixed checkpoint"),
    ],
)
def test_malformed_checkpoints_are_rejected(tmp_path, payload, message):
    path = tmp_path / "bad.ckpt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match=message):
        load_model_weights(path)


def test_checkpoint_loader_does_not_fall_back_to_unrestricted_pickle(tmp_path):
    path = tmp_path / "objects.ckpt"
    torch.save({"state_dict": {"weight": torch.ones(1)}, "extra": SimpleNamespace(value=1)}, path)
    with pytest.raises(ValueError, match="tensor-only loader"):
        load_model_weights(path)


def test_unknown_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError, match=".ckpt or .safetensors"):
        load_model_weights(tmp_path / "weights.pt")


def test_export_handles_shared_noncontiguous_tensors_and_metadata(tmp_path):
    model = torch.nn.Module()
    model.register_buffer("matrix", torch.arange(6).reshape(2, 3).t())
    model.register_buffer("alias", model.matrix)
    path = tmp_path / "weights.safetensors"
    save_model_safetensors(model, path, {"global_step": "7"})
    weights = load_model_weights(path)
    assert torch.equal(weights["matrix"], model.matrix)
    assert torch.equal(weights["alias"], model.alias)
    with safe_open(path, framework="pt") as reader:
        assert reader.metadata() == {"global_step": "7"}


def test_failed_atomic_export_preserves_previous_file_and_cleans_temporary_file(tmp_path, monkeypatch):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"previous export")

    def fail(*args, **kwargs):
        raise OSError("simulated filesystem failure")

    monkeypatch.setattr("safetensors.torch.save_file", fail)
    with pytest.raises(OSError, match="simulated filesystem failure"):
        save_model_safetensors(torch.nn.Linear(1, 1), path)
    assert path.read_bytes() == b"previous export"
    assert list(tmp_path.iterdir()) == [path]


class _Agent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.build_target_encoder = True
        self.config = LatentWAMConfig()
        self.model = torch.nn.Linear(1, 1, bias=False)


class _Module(LatentWAMLightningModule):
    def training_step(self, batch, batch_idx):
        self.log("score", float(self.global_step), logger=False)
        return self.agent.model(batch[0]).square().sum()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def on_train_batch_end(self, outputs, batch, batch_idx):
        pass


def _trainer(tmp_path, callback, max_steps=2, extra_callbacks=()):
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=max_steps,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[callback, *extra_callbacks],
        default_root_dir=tmp_path,
    )


def _checkpoint_options(tmp_path):
    return dict(dirpath=tmp_path, every_n_train_steps=1, save_top_k=1, monitor="score", mode="max")


@pytest.mark.parametrize("enabled", [False, True])
def test_real_training_saves_full_checkpoint_and_optional_paired_export(tmp_path, enabled):
    callback = LatentWAMModelCheckpoint(export_safetensors=enabled, **_checkpoint_options(tmp_path))
    assert callback.state_key == pl.callbacks.ModelCheckpoint(**_checkpoint_options(tmp_path)).state_key
    module = _Module(_Agent())
    trainer = _trainer(tmp_path, callback)
    trainer.fit(module, train_dataloaders=DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1))
    checkpoints = list(tmp_path.glob("*.ckpt"))
    exports = list(tmp_path.glob("*.safetensors"))
    assert len(checkpoints) == 1  # save_top_k prunes both members of the old pair.
    assert len(exports) == int(enabled)
    checkpoint = torch.load(checkpoints[0], weights_only=True)
    assert checkpoint["optimizer_states"][0]["state"]
    assert checkpoint["lr_schedulers"][0]["last_epoch"] == 2
    assert checkpoint["global_step"] == 2
    if enabled:
        expected = load_model_weights(checkpoints[0])
        exported = load_model_weights(exports[0])
        assert expected.keys() == exported.keys()
        assert all(torch.equal(expected[key], exported[key]) for key in expected)
        with safe_open(exports[0], framework="pt") as reader:
            metadata = reader.metadata()
        assert metadata["global_step"] == "2"
        assert metadata["source_checkpoint"] == checkpoints[0].name
        assert metadata[TEMPORAL_ATTENTION_MASK_CHECKPOINT_KEY] == "block_diagonal"


def test_exact_resume_restores_original_callback_optimizer_and_scheduler_state(tmp_path):
    options = _checkpoint_options(tmp_path)
    old_callback = pl.callbacks.ModelCheckpoint(**options)
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    old_trainer = _trainer(tmp_path, old_callback, max_steps=1)
    old_trainer.fit(_Module(_Agent()), train_dataloaders=loader)
    path = old_callback.best_model_path
    new_callback = LatentWAMModelCheckpoint(export_safetensors=True, **options)
    observed = []

    class Probe(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == 1
            assert trainer.optimizers[0].state
            assert trainer.lr_scheduler_configs[0].scheduler.last_epoch == 1
            assert torch.equal(new_callback.best_model_score, old_callback.best_model_score)
            observed.append(True)

    trainer = _trainer(tmp_path, new_callback, extra_callbacks=[Probe()])
    trainer.fit(_Module(_Agent()), train_dataloaders=loader, ckpt_path=path)
    assert observed == [True]
    assert trainer.global_step == 2
    assert trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.025)


def test_nonzero_rank_does_not_export(tmp_path, monkeypatch):
    export = Mock()
    monkeypatch.setattr("navsim.agents.latent_wam.checkpoint.save_model_safetensors", export)
    trainer = SimpleNamespace(save_checkpoint=Mock(), global_step=1, is_global_zero=False)
    LatentWAMModelCheckpoint(export_safetensors=True)._save_checkpoint(trainer, str(tmp_path / "rank.ckpt"))
    trainer.save_checkpoint.assert_called_once()
    export.assert_not_called()


def test_export_failure_keeps_full_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / "full.ckpt"
    model = _Module(_Agent())
    trainer = SimpleNamespace(
        save_checkpoint=lambda filepath, *args: path.write_bytes(b"full training state"),
        global_step=1,
        current_epoch=0,
        is_global_zero=True,
        loggers=[],
        lightning_module=model,
    )
    monkeypatch.setattr("safetensors.torch.save_file", Mock(side_effect=OSError("export failure")))
    with pytest.raises(OSError, match="export failure"):
        LatentWAMModelCheckpoint(export_safetensors=True)._save_checkpoint(trainer, str(path))
    assert path.read_bytes() == b"full training state"
    assert not path.with_suffix(".safetensors").exists()


def test_full_training_state_cannot_be_disabled():
    with pytest.raises(ValueError, match="full training state"):
        LatentWAMModelCheckpoint(save_weights_only=True)
