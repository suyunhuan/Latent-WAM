"""Manual conversion preserves source files and never restores training state."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from safetensors import safe_open

from navsim.agents.latent_wam.checkpoint import convert_checkpoint_to_safetensors, load_model_weights
from navsim.agents.latent_wam.config import TrainingConfig

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/convert_latent_wam_checkpoint.py"


def _source(path, prefix="agent.model.", legacy=False):
    shared = torch.arange(6).reshape(2, 3).t()
    checkpoint = {
        "state_dict": {
            prefix + "weight": shared,
            prefix + "alias": shared,
            prefix + "target_dino_encoder.weight": torch.ones(2),
        },
        "optimizer_states": [{"state": {0: {"step": torch.tensor(17)}}}],
        "lr_schedulers": [{"last_epoch": 17}],
        "epoch": 2,
        "global_step": 17,
    }
    if not legacy:
        checkpoint["latent_wam_temporal_attention_mask"] = "block_causal"
    torch.save(checkpoint, path)


@pytest.mark.parametrize("prefix", ["", "agent.model."])
@pytest.mark.parametrize("inference_only", [False, True])
def test_conversion_is_lossless_with_explicit_ema_policy(tmp_path, prefix, inference_only):
    source = tmp_path / "source.ckpt"
    destination = tmp_path / "converted.safetensors"
    _source(source, prefix)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = convert_checkpoint_to_safetensors(source, destination, inference_only=inference_only)
    expected = load_model_weights(source)
    if inference_only:
        expected = {key: value for key, value in expected.items() if not key.startswith("target_dino_encoder.")}
    result = load_model_weights(destination)
    assert result.keys() == expected.keys()
    assert all(torch.equal(result[key], value) for key, value in expected.items())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with safe_open(destination, framework="pt") as reader:
        assert reader.metadata() == metadata
    assert metadata["global_step"] == "17"
    assert metadata["epoch"] == "2"
    assert metadata["source_checkpoint"] == source.name
    assert metadata["latent_wam_temporal_attention_mask"] == "block_causal"
    assert metadata["ema_target_encoder"] == ("excluded" if inference_only else "included")


def test_legacy_checkpoint_records_diagonal_mode(tmp_path):
    source, destination = tmp_path / "old.ckpt", tmp_path / "old.safetensors"
    _source(source, legacy=True)
    metadata = convert_checkpoint_to_safetensors(source, destination)
    assert metadata["latent_wam_temporal_attention_mask"] == "block_diagonal"


def test_existing_output_requires_explicit_overwrite(tmp_path):
    source, destination = tmp_path / "old.ckpt", tmp_path / "old.safetensors"
    _source(source)
    destination.write_bytes(b"retained output")
    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_checkpoint_to_safetensors(source, destination)
    assert destination.read_bytes() == b"retained output"
    convert_checkpoint_to_safetensors(source, destination, overwrite=True)
    assert "weight" in load_model_weights(destination)


def test_publication_race_does_not_replace_another_output(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.ckpt", tmp_path / "output.safetensors"
    _source(source)
    link = os.link

    def competing_writer(temporary, output):
        destination.write_bytes(b"other writer")
        link(temporary, output)

    monkeypatch.setattr("navsim.agents.latent_wam.checkpoint.os.link", competing_writer)
    with pytest.raises(FileExistsError):
        convert_checkpoint_to_safetensors(source, destination)
    assert destination.read_bytes() == b"other writer"
    assert not list(tmp_path.glob(".*.tmp"))


def test_input_alias_cannot_be_overwritten(tmp_path):
    source, destination = tmp_path / "source.ckpt", tmp_path / "alias.safetensors"
    _source(source)
    os.link(source, destination)
    before = source.read_bytes()
    with pytest.raises(ValueError, match="different files"):
        convert_checkpoint_to_safetensors(source, destination, overwrite=True)
    assert source.read_bytes() == before


@pytest.mark.parametrize("kind", ["extension", "mixed", "mask", "missing", "only_target"])
def test_invalid_conversion_leaves_no_output(tmp_path, kind):
    source, destination = tmp_path / "source.ckpt", tmp_path / "output.safetensors"
    _source(source)
    checkpoint = torch.load(source, weights_only=True)
    if kind == "extension":
        destination = destination.with_suffix(".pt")
    elif kind == "mixed":
        checkpoint["state_dict"]["extra"] = torch.ones(1)
    elif kind == "mask":
        checkpoint["latent_wam_temporal_attention_mask"] = "unknown"
    elif kind == "missing":
        del checkpoint["state_dict"]
    elif kind == "only_target":
        checkpoint["state_dict"] = {"agent.model.target_dino_encoder.weight": torch.ones(1)}
    torch.save(checkpoint, source)
    with pytest.raises(ValueError):
        convert_checkpoint_to_safetensors(source, destination, inference_only=kind == "only_target")
    assert not destination.exists()


def test_cli_default_destination_collision_and_overwrite(tmp_path):
    source = tmp_path / "old run.ckpt"
    _source(source)
    command = [sys.executable, str(SCRIPT), str(source)]
    first = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert first.returncode == 0, first.stderr
    destination = source.with_suffix(".safetensors")
    assert json.loads(first.stdout)["output"] == str(destination)
    assert "target_dino_encoder.weight" in load_model_weights(destination)
    second = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert second.returncode != 0 and "--overwrite" in second.stderr
    third = subprocess.run(
        [*command, "--overwrite", "--inference-only"], cwd=tmp_path, text=True, capture_output=True, timeout=30
    )
    assert third.returncode == 0, third.stderr
    assert "target_dino_encoder.weight" not in load_model_weights(destination)


def test_cli_explicit_destination_and_help(tmp_path):
    source, destination = tmp_path / "old.ckpt", tmp_path / "nested folder" / "weights.safetensors"
    _source(source)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(source), "--output", str(destination)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert destination.is_file()
    help_result = subprocess.run([sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True, timeout=10)
    assert help_result.returncode == 0 and "--inference-only" in help_result.stdout


def test_training_defaults_to_ckpt_only():
    assert TrainingConfig().export_safetensors is False
    cfg = OmegaConf.load(ROOT / "navsim/planning/script/config/common/agent/latent_wam_agent.yaml")
    assert cfg.config.training.export_safetensors is False
