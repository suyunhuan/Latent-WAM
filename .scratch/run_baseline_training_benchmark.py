"""Run the pre-refactor benchmark without loading an unavailable agent checkpoint."""

from __future__ import annotations

import runpy

from navsim.agents.latent_wam.agent import LatentWAMAgent


def _skip_checkpoint_loading(self: LatentWAMAgent) -> None:
    """Keep random initialization, matching a training run without a resume checkpoint."""


LatentWAMAgent.initialize = _skip_checkpoint_loading
runpy.run_path("/tmp/world4drive-baseline-head/scripts/benchmark_latent_wam_training.py", run_name="__main__")
