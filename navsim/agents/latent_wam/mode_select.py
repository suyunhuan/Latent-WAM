"""Inference-time trajectory mode selection (command vs mode_logits).

Training still uses WTA on ``all_trajectories``; released inference defaults to
selecting the mode index from the ego driving-command one-hot. This module
lets evaluation override that choice without changing the default path.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

MODE_SELECT_COMMAND = "command"
MODE_SELECT_MODE_LOGITS = "mode_logits"
MODE_SELECT_HYBRID = "hybrid"
MODE_SELECT_CHOICES = (MODE_SELECT_COMMAND, MODE_SELECT_MODE_LOGITS, MODE_SELECT_HYBRID)

ENV_MODE_SELECT = "LATENT_WAM_MODE_SELECT"
ENV_HYBRID_MARGIN = "LATENT_WAM_HYBRID_MARGIN"


def resolve_mode_select(configured: str = MODE_SELECT_COMMAND) -> str:
    """Resolve strategy from env override, then config; default is command."""
    env = os.environ.get(ENV_MODE_SELECT, "").strip().lower()
    strategy = env or str(configured or MODE_SELECT_COMMAND).strip().lower()
    if strategy not in MODE_SELECT_CHOICES:
        raise ValueError(f"mode_select must be one of {MODE_SELECT_CHOICES}, got {strategy!r}")
    return strategy


def resolve_hybrid_margin(configured: float = 0.0) -> float:
    """Resolve hybrid margin from env override, then config."""
    env = os.environ.get(ENV_HYBRID_MARGIN, "").strip()
    if env:
        return float(env)
    return float(configured)


def select_mode_indices(
    mode_logits: torch.Tensor,
    command: torch.Tensor,
    strategy: str,
    *,
    hybrid_margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pick per-sample mode indices and a disagreement mask vs command.

    Args:
        mode_logits: ``[B, num_modes]`` scores from the trajectory head.
        command: ``[B]`` integer command indices (0..num_modes-1).
        strategy: One of :data:`MODE_SELECT_CHOICES`.
        hybrid_margin: For ``hybrid``, override to ``mode_logits.argmax`` only when
            ``logit[best] - logit[command] > hybrid_margin``. Margin ``0`` means
            override on any disagreement (equivalent to ``mode_logits`` for the
            selected index when logits differ).

    Returns:
        ``(selected_indices, disagreement_mask)`` both shape ``[B]``.
    """
    if mode_logits.ndim != 2:
        raise ValueError(f"mode_logits must have shape [B, M], got {tuple(mode_logits.shape)}")
    command = command.long().reshape(-1)
    if command.shape[0] != mode_logits.shape[0]:
        raise ValueError("command batch size must match mode_logits")
    if strategy not in MODE_SELECT_CHOICES:
        raise ValueError(f"mode_select must be one of {MODE_SELECT_CHOICES}, got {strategy!r}")

    logits_idx = mode_logits.argmax(dim=-1)
    disagree = logits_idx != command
    if strategy == MODE_SELECT_COMMAND:
        return command, disagree
    if strategy == MODE_SELECT_MODE_LOGITS:
        return logits_idx, disagree

    batch = torch.arange(mode_logits.shape[0], device=mode_logits.device)
    margin = mode_logits[batch, logits_idx] - mode_logits[batch, command]
    selected = torch.where(margin > hybrid_margin, logits_idx, command)
    return selected, disagree


def apply_mode_selection(
    prediction: dict,
    command: torch.Tensor,
    strategy: str,
    *,
    hybrid_margin: float = 0.0,
) -> Tuple[dict, torch.Tensor]:
    """Rewrite ``prediction['trajectory']`` according to ``strategy``.

    Returns the (possibly updated) prediction dict and the disagreement mask.
    """
    mode_logits = prediction["mode_logits"]
    all_trajectories = prediction["all_trajectories"]
    selected, disagree = select_mode_indices(
        mode_logits, command, strategy, hybrid_margin=hybrid_margin
    )
    if strategy != MODE_SELECT_COMMAND:
        batch = torch.arange(all_trajectories.shape[0], device=all_trajectories.device)
        prediction = dict(prediction)
        prediction["trajectory"] = all_trajectories[batch, selected]
    return prediction, disagree
