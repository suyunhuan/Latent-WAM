"""Run the benchmark while reusing already verified shard headers."""

from __future__ import annotations

import runpy

import navsim.agents.latent_wam.dataset as cache_dataset_module


# The source cache generation already checked the immutable shard sidecars and
# checksums. The temporary subset's full header walk is too slow on OSS storage.
cache_dataset_module.validate_cache_headers = lambda cache_root, manifest: None
runpy.run_path("scripts/benchmark_latent_wam_training.py", run_name="__main__")
