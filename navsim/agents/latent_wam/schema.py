"""Canonical identifiers for the only supported Latent WAM cache format."""

from navsim.agents.latent_wam.cache import CACHE_FORMAT, SCHEMA_REVISION

FEATURE_KEY = "latent_wam_features"
TARGET_KEY = "latent_wam_targets"

__all__ = ["CACHE_FORMAT", "SCHEMA_REVISION", "FEATURE_KEY", "TARGET_KEY"]
