import os

import pytest
import torch

from navsim.agents.latent_wam.config import LatentWAMConfig, RuntimeConfig
from navsim.agents.latent_wam.mode_select import (
    apply_mode_selection,
    resolve_mode_select,
    select_mode_indices,
)


def test_select_mode_indices_command_and_logits() -> None:
    mode_logits = torch.tensor([[0.1, 0.9, 0.0, 0.0], [0.8, 0.1, 0.05, 0.05]])
    command = torch.tensor([0, 0])

    cmd_idx, disagree = select_mode_indices(mode_logits, command, "command")
    assert torch.equal(cmd_idx, command)
    assert torch.equal(disagree, torch.tensor([True, False]))

    logits_idx, disagree2 = select_mode_indices(mode_logits, command, "mode_logits")
    assert torch.equal(logits_idx, torch.tensor([1, 0]))
    assert torch.equal(disagree2, disagree)


def test_hybrid_margin_keeps_command_when_small() -> None:
    mode_logits = torch.tensor([[1.0, 1.1, 0.0, 0.0]])
    command = torch.tensor([0])
    selected, _ = select_mode_indices(mode_logits, command, "hybrid", hybrid_margin=0.2)
    assert torch.equal(selected, command)
    selected2, _ = select_mode_indices(mode_logits, command, "hybrid", hybrid_margin=0.0)
    assert torch.equal(selected2, torch.tensor([1]))


def test_apply_mode_selection_rewrites_trajectory() -> None:
    all_trajectories = torch.zeros((1, 4, 8, 3))
    all_trajectories[0, 2] = 7.0
    prediction = {
        "trajectory": all_trajectories[0, 0].clone(),
        "all_trajectories": all_trajectories,
        "mode_logits": torch.tensor([[0.0, 0.0, 3.0, 0.0]]),
    }
    updated, disagree = apply_mode_selection(
        prediction, torch.tensor([0]), "mode_logits"
    )
    assert bool(disagree.item())
    assert torch.allclose(updated["trajectory"], torch.full((8, 3), 7.0))
    # command path leaves trajectory untouched
    kept, _ = apply_mode_selection(prediction, torch.tensor([0]), "command")
    assert torch.allclose(kept["trajectory"], prediction["trajectory"])


def test_env_overrides_configured_mode_select(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LATENT_WAM_MODE_SELECT", raising=False)
    assert resolve_mode_select("command") == "command"
    monkeypatch.setenv("LATENT_WAM_MODE_SELECT", "mode_logits")
    assert resolve_mode_select("command") == "mode_logits"


def test_runtime_config_rejects_invalid_mode_select() -> None:
    with pytest.raises(ValueError, match="mode_select"):
        LatentWAMConfig(runtime=RuntimeConfig(mode_select="nope"))


def test_default_runtime_mode_select_is_command() -> None:
    assert LatentWAMConfig().runtime.mode_select == "command"
