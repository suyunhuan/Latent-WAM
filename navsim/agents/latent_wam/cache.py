"""Versioned feature schema and legacy cache adapters for Latent WAM."""

import gzip
import os
import pickle
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np
import torch

from navsim.agents.latent_wam.schema import (
    CACHE_GENERATION_KEY,
    CACHE_SCHEMA_VERSION,
    PUBLIC_FEATURE_NAME,
    PUBLIC_GEOMETRY_TOKEN_COUNT,
    PUBLIC_TARGET_NAME,
)

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
    schema_version: int = CACHE_SCHEMA_VERSION

    def validate(self) -> None:
        """Validate tensor ranks, time alignment, and finite values."""
        if self.schema_version != CACHE_SCHEMA_VERSION:
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


def public_features_from_mapping(data: Mapping[str, Any]) -> LatentWAMFeatures:
    """Load and validate public features from their serialized mapping.

    Args:
        data: Deserialized public feature payload.

    Returns:
        Validated Latent WAM feature tensors.

    Raises:
        ValueError: If the schema or tensor shapes are invalid.
    """
    if "schema_version" not in data:
        raise ValueError("public cache is missing schema_version")
    features = LatentWAMFeatures(
        images=torch.as_tensor(data["images"]),
        geometry=torch.as_tensor(data["geometry"]),
        ego_status=torch.as_tensor(data["ego_status"], dtype=torch.float32),
        time_offsets_s=tuple(data.get("time_offsets_s", TIME_OFFSETS_S)),
        schema_version=int(data["schema_version"]),
    )
    features.validate()
    if features.images.dtype != torch.uint8:
        raise ValueError("public cache images must use uint8")
    if features.geometry.shape[2] != PUBLIC_GEOMETRY_TOKEN_COUNT:
        raise ValueError(f"public cache geometry must contain {PUBLIC_GEOMETRY_TOKEN_COUNT} tokens")
    if features.geometry.dtype != torch.float32:
        raise ValueError("public cache geometry must use float32")
    return features


def public_trajectory_from_mapping(data: Mapping[str, Any], expected_num_poses: int) -> torch.Tensor:
    """Load and validate the public trajectory target.

    Args:
        data: Deserialized public target payload.
        expected_num_poses: Required trajectory length.

    Returns:
        Float32 trajectory tensor shaped ``[expected_num_poses, 3]``.

    Raises:
        ValueError: If the target is missing, malformed, or non-finite.
    """
    if "trajectory" not in data:
        raise ValueError("public target cache is missing trajectory")
    trajectory = torch.as_tensor(data["trajectory"], dtype=torch.float32)
    if trajectory.shape != (expected_num_poses, 3):
        raise ValueError(f"trajectory must have shape [{expected_num_poses},3]")
    if not torch.isfinite(trajectory).all():
        raise ValueError("trajectory must contain finite values")
    return trajectory


def is_valid_public_cache(feature_path: Path, target_path: Path, expected_num_poses: int) -> bool:
    """Check whether a public feature/target pair passes full validation.

    Args:
        feature_path: Gzip-pickle feature file.
        target_path: Gzip-pickle target file.
        expected_num_poses: Required trajectory length.

    Returns:
        ``True`` only when both files can be fully loaded and validated.
    """
    if not feature_path.is_file() or not target_path.is_file():
        return False
    try:
        feature_data = load_trusted_pickle(feature_path)
        target_data = load_trusted_pickle(target_path)
        feature_generation = feature_data.get(CACHE_GENERATION_KEY)
        target_generation = target_data.get(CACHE_GENERATION_KEY)
        if not isinstance(feature_generation, str) or not feature_generation or feature_generation != target_generation:
            return False
        public_features_from_mapping(feature_data)
        public_trajectory_from_mapping(target_data, expected_num_poses)
    except Exception:
        return False
    return True


def _dump_pickle(path: Path, data: Mapping[str, Any]) -> None:
    with gzip.open(path, "wb", compresslevel=1) as file:
        pickle.dump(dict(data), file)


def publish_public_cache(
    sample_root: Path,
    features: Mapping[str, Any],
    targets: Mapping[str, Any],
    rank: int,
    expected_num_poses: int,
) -> None:
    """Validate and atomically publish one public v2 feature/target pair.

    Args:
        sample_root: Token-specific cache directory.
        features: Public feature payload to serialize.
        targets: Public target payload to serialize.
        rank: Cache worker rank used to isolate temporary files.
        expected_num_poses: Required trajectory length for reload validation.

    Raises:
        ValueError: If serialized temporary files fail full validation.
    """
    sample_root.mkdir(parents=True, exist_ok=True)
    suffix = f".rank_{rank}.pid_{os.getpid()}.tmp"
    feature_path = sample_root / PUBLIC_FEATURE_NAME
    target_path = sample_root / PUBLIC_TARGET_NAME
    temporary_feature = feature_path.with_name(feature_path.name + suffix)
    temporary_target = target_path.with_name(target_path.name + suffix)
    generation = uuid.uuid4().hex
    feature_payload = {**features, CACHE_GENERATION_KEY: generation}
    target_payload = {**targets, CACHE_GENERATION_KEY: generation}
    try:
        _dump_pickle(temporary_feature, feature_payload)
        _dump_pickle(temporary_target, target_payload)
        if not is_valid_public_cache(temporary_feature, temporary_target, expected_num_poses):
            raise ValueError("temporary public cache failed validation")
        os.replace(temporary_feature, feature_path)
        os.replace(temporary_target, target_path)
    finally:
        temporary_feature.unlink(missing_ok=True)
        temporary_target.unlink(missing_ok=True)


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
