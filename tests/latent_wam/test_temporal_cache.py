"""Changing world-model attention does not change cached input semantics."""

from dataclasses import asdict
from types import SimpleNamespace

from omegaconf import OmegaConf

from navsim.agents.latent_wam.config import LatentWAMConfig, TrainingConfig
from navsim.planning.script.run_latent_wam_caching import _cache_fingerprint


def test_attention_modes_have_identical_cache_fingerprints() -> None:
    plan = SimpleNamespace(samples=[SimpleNamespace(log_name="test-log")])
    fingerprints = []
    for mode in ("block_diagonal", "block_causal"):
        config = LatentWAMConfig(training=TrainingConfig(temporal_attention_mask=mode))
        cfg = OmegaConf.create({"train_test_split": {"data_split": "trainval"}, "agent": {"config": asdict(config)}})
        fingerprints.append(_cache_fingerprint(cfg, plan))
    assert fingerprints[0] == fingerprints[1]
