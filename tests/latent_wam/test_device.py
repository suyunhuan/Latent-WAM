"""Unit tests for Latent WAM device selection."""

from __future__ import annotations

import torch

from navsim.agents.latent_wam.device import (
    lightning_accelerator,
    resolve_torch_device,
    training_precision,
)


def test_auto_prefers_cuda_when_available(monkeypatch):
    monkeypatch.delenv("LATENT_WAM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("navsim.agents.latent_wam.device.mps_is_available", lambda: True)
    device = resolve_torch_device()
    assert device.type == "cuda"


def test_auto_uses_mps_without_cuda(monkeypatch):
    monkeypatch.delenv("LATENT_WAM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("navsim.agents.latent_wam.device.mps_is_available", lambda: True)
    device = resolve_torch_device()
    assert device.type == "mps"
    assert lightning_accelerator(device) == "mps"
    assert training_precision("bf16-mixed") == "32-true"


def test_env_override_cpu(monkeypatch):
    monkeypatch.setenv("LATENT_WAM_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("navsim.agents.latent_wam.device.mps_is_available", lambda: True)
    assert resolve_torch_device().type == "cpu"


def test_require_accelerator_rejects_cpu_only_auto(monkeypatch):
    monkeypatch.delenv("LATENT_WAM_DEVICE", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr("navsim.agents.latent_wam.device.mps_is_available", lambda: False)
    try:
        resolve_torch_device(require_accelerator=True)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
