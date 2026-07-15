"""Temporary adapter for benchmarking the removed sample-gzip cache reader."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Dict, Tuple

import torch


class LegacySampleCacheDataset(torch.utils.data.Dataset):
    """Read trusted v2 sample-gzip cache files for a one-off comparison."""

    def __init__(self, cache_path: str | Path) -> None:
        self.cache_path = Path(cache_path)
        self.samples = sorted(
            (feature_path, feature_path.with_name("latent_wam_targets_v2.gz"))
            for feature_path in self.cache_path.glob("*/*/latent_wam_features_v2.gz")
            if feature_path.with_name("latent_wam_targets_v2.gz").is_file()
        )
        if not self.samples:
            raise ValueError(f"sample-gzip cache is empty: {self.cache_path}")

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load(path: Path) -> Dict[str, object]:
        with gzip.open(path, "rb") as file:
            value = pickle.load(file)
        if not isinstance(value, dict):
            raise ValueError(f"cache payload is not a mapping: {path}")
        return value

    def __getitem__(self, index: int) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
        feature_path, target_path = self.samples[index]
        features = self._load(feature_path)
        targets = self._load(target_path)
        return (
            {
                "images": torch.as_tensor(features["images"]),
                "geometry": torch.as_tensor(features["geometry"], dtype=torch.float32),
                "ego_status": torch.as_tensor(features["ego_status"], dtype=torch.float32),
                "token": feature_path.parent.name,
            },
            {"trajectory": torch.as_tensor(targets["trajectory"], dtype=torch.float32)},
        )
