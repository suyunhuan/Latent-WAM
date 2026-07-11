"""Datasets for the versioned and legacy Latent WAM feature caches."""

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch

from navsim.agents.latent_wam.cache import (
    load_legacy_features,
    load_trusted_pickle,
    public_features_from_mapping,
    public_trajectory_from_mapping,
)
from navsim.agents.latent_wam.schema import PUBLIC_FEATURE_NAME, PUBLIC_TARGET_NAME

LEGACY_FEATURE_NAME = "transfuser_feature.gz"
LEGACY_TARGET_NAME = "transfuser_target.gz"


class LatentWAMCacheDataset(torch.utils.data.Dataset):
    """Read the public cache schema and the original two-level cache layout."""

    def __init__(
        self,
        cache_path: str | Path,
        log_names: Optional[Iterable[str]] = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_dir():
            raise ValueError(f"cache path does not exist: {self.cache_path}")
        allowed_logs = set(log_names) if log_names is not None else None
        samples_by_root: Dict[Path, Tuple[Path, Path]] = {}
        for feature_path in self.cache_path.glob(f"*/*/{PUBLIC_FEATURE_NAME}"):
            if allowed_logs is None or feature_path.parent.parent.name in allowed_logs:
                target_path = feature_path.with_name(PUBLIC_TARGET_NAME)
                if target_path.is_file():
                    samples_by_root[feature_path.parent] = (feature_path, target_path)
        for feature_path in self.cache_path.glob(f"*/*/{LEGACY_FEATURE_NAME}"):
            if allowed_logs is None or feature_path.parent.parent.name in allowed_logs:
                target_path = feature_path.with_name(LEGACY_TARGET_NAME)
                if target_path.is_file() and feature_path.parent not in samples_by_root:
                    samples_by_root[feature_path.parent] = (feature_path, target_path)
        self.samples = sorted(samples_by_root.values())

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
        feature_path, target_path = self.samples[index]
        if feature_path.name == LEGACY_FEATURE_NAME:
            features = load_legacy_features(feature_path, self.cache_path / "feature_cache")
        else:
            features = public_features_from_mapping(load_trusted_pickle(feature_path))
        target_data = load_trusted_pickle(target_path)
        trajectory = public_trajectory_from_mapping(target_data, expected_num_poses=8)
        return (
            {
                "images": features.images,
                "geometry": features.geometry,
                "ego_status": features.ego_status,
                "token": feature_path.parent.name,
            },
            {"trajectory": trajectory},
        )
