"""Versioned feature schema and legacy cache adapters for Latent WAM."""

import gzip
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

LEGACY_FRAME_KEYS: Tuple[str, ...] = (
    "camera_feature_prev_3",
    "camera_feature",
    "camera_feature_next_4",
    "camera_feature_next_8",
)
LEGACY_STATUS_KEYS: Tuple[str, ...] = (
    "status_feature_prev_3",
    "status_feature",
    "status_feature_next_4",
    "status_feature_next_8",
)
TIME_OFFSETS_S: Tuple[float, ...] = (-1.5, 0.0, 2.0, 4.0)


@dataclass(frozen=True)
class LatentWAMFeatures:
    """In-memory feature batch before collation.

    Images have shape ``[T, V, H, W, C]``, geometry features have shape
    ``[T, V, N, D]``, and ego status has shape ``[T, 8]``.
    """

    images: torch.Tensor
    geometry: torch.Tensor
    ego_status: torch.Tensor
    time_offsets_s: Tuple[float, ...] = TIME_OFFSETS_S
    schema_version: int = 1

    def validate(self) -> None:
        """Validate tensor ranks, time alignment, and finite values."""
        if self.schema_version != 1:
            raise ValueError(f"unsupported cache schema version: {self.schema_version}")
        if self.time_offsets_s != TIME_OFFSETS_S:
            raise ValueError(f"cache time offsets must be {TIME_OFFSETS_S}")
        num_frames = len(self.time_offsets_s)
        if self.images.ndim != 5 or self.images.shape[0] != num_frames:
            raise ValueError("images must have shape [T, V, H, W, C]")
        if self.geometry.ndim != 4 or self.geometry.shape[0] != num_frames:
            raise ValueError("geometry must have shape [T, V, N, D]")
        if self.ego_status.shape != (num_frames, 8):
            raise ValueError("ego_status must have shape [T, 8]")
        if self.images.shape[1] != self.geometry.shape[1]:
            raise ValueError("images and geometry must contain the same number of views")
        if self.images.shape[1] != 3 or self.images.shape[-1] != 3:
            raise ValueError("images must contain three RGB camera views")
        if self.geometry.shape[-1] != 2048:
            raise ValueError("geometry features must have dimension 2048")
        if not torch.isfinite(self.geometry).all() or not torch.isfinite(self.ego_status).all():
            raise ValueError("cached features must contain finite values")


def load_trusted_pickle(path: Path) -> Any:
    """Load a trusted legacy gzip-pickle file.

    Legacy caches use pickle and must never be loaded from an untrusted source.
    """
    with gzip.open(path, "rb") as file:
        return pickle.load(file)


def _load_legacy_frame(cache_root: Path, token: str) -> Tuple[torch.Tensor, torch.Tensor]:
    frame_root = cache_root / token
    frame = load_trusted_pickle(frame_root / f"{token}.gz")
    image = torch.as_tensor(frame["dino_feature"])
    geometry_path = frame["geometry_feature"]
    geometry_path = Path(geometry_path)
    if not geometry_path.is_absolute() or not geometry_path.exists():
        geometry_path = frame_root / "geometry_feature.npy"
    geometry = torch.from_numpy(np.array(np.load(geometry_path, mmap_mode="r"), copy=True))
    if geometry.ndim == 4 and geometry.shape[0] == 1:
        geometry = geometry.squeeze(0)
    return image, geometry


def load_legacy_features(index_path: Path, cache_root: Path) -> LatentWAMFeatures:
    """Load one complete legacy cache index into the versioned schema."""
    index: Mapping[str, Any] = load_trusted_pickle(index_path)
    missing = [key for key in (*LEGACY_FRAME_KEYS, *LEGACY_STATUS_KEYS) if key not in index]
    if missing:
        raise ValueError(f"legacy cache index is missing keys: {missing}")

    frames = [_load_legacy_frame(cache_root, str(index[key])) for key in LEGACY_FRAME_KEYS]
    images = torch.stack([frame[0] for frame in frames])
    geometry = torch.stack([frame[1] for frame in frames])
    ego_status = torch.stack([torch.as_tensor(index[key], dtype=torch.float32) for key in LEGACY_STATUS_KEYS])
    features = LatentWAMFeatures(images=images, geometry=geometry, ego_status=ego_status)
    features.validate()
    return features


def build_structural_fixture(
    index_path: Path,
    frame_cache_path: Path,
    geometry_path: Path,
) -> LatentWAMFeatures:
    """Build a four-frame structural fixture from one real cached frame."""
    index: Mapping[str, Any] = load_trusted_pickle(index_path)
    frame: Dict[str, Any] = load_trusted_pickle(frame_cache_path)
    image = torch.as_tensor(frame["dino_feature"])
    geometry = torch.from_numpy(np.array(np.load(geometry_path, mmap_mode="r"), copy=True))
    if geometry.ndim == 4 and geometry.shape[0] == 1:
        geometry = geometry.squeeze(0)
    ego_status = torch.stack([torch.as_tensor(index[key], dtype=torch.float32) for key in LEGACY_STATUS_KEYS])
    features = LatentWAMFeatures(
        images=torch.stack([image] * len(TIME_OFFSETS_S)),
        geometry=torch.stack([geometry] * len(TIME_OFFSETS_S)),
        ego_status=ego_status,
    )
    features.validate()
    return features


def to_legacy_model_input(features: LatentWAMFeatures) -> Dict[str, Any]:
    """Convert versioned features to the dictionary accepted by the released model."""
    features.validate()
    result: Dict[str, Any] = {}
    for frame_idx, key in enumerate(LEGACY_FRAME_KEYS):
        result[key] = {
            "dino_feature": features.images[frame_idx],
            "geometry_feature": features.geometry[frame_idx].unsqueeze(0),
        }
    for frame_idx, key in enumerate(LEGACY_STATUS_KEYS):
        result[key] = features.ego_status[frame_idx]
    return result
