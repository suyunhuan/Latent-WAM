"""Run the standard benchmark with the temporary sample-gzip dataset adapter."""

from __future__ import annotations

import runpy

import navsim.agents.latent_wam.dataset as dataset_module

from benchmark_legacy_sample_cache import LegacySampleCacheDataset

dataset_module.LatentWAMCacheDataset = LegacySampleCacheDataset
runpy.run_path("scripts/benchmark_latent_wam_training.py", run_name="__main__")
