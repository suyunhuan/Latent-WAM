import gzip
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from navsim.agents.latent_wam.cache import (
    LEGACY_FRAME_KEYS,
    LEGACY_STATUS_KEYS,
    build_structural_fixture,
    is_valid_public_cache,
    load_legacy_features,
    publish_public_cache,
)
from navsim.agents.latent_wam.dataset import LatentWAMCacheDataset
from navsim.agents.latent_wam.schema import PUBLIC_FEATURE_NAME, PUBLIC_TARGET_NAME


def _dump_pickle(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as file:
        pickle.dump(value, file)


def _write_frame(cache_root: Path, token: str) -> None:
    frame_root = cache_root / token
    frame_root.mkdir(parents=True)
    geometry_path = frame_root / "geometry_feature.npy"
    np.save(geometry_path, np.ones((1, 3, 9, 2048), dtype=np.float32))
    _dump_pickle(
        frame_root / f"{token}.gz",
        {
            "dino_feature": np.zeros((3, 8, 12, 3), dtype=np.uint8),
            "geometry_feature": geometry_path,
        },
    )


def test_load_legacy_features(tmp_path: Path) -> None:
    index = {}
    for frame_idx, frame_key in enumerate(LEGACY_FRAME_KEYS):
        token = f"frame-{frame_idx}"
        index[frame_key] = token
        _write_frame(tmp_path, token)
    for status_idx, status_key in enumerate(LEGACY_STATUS_KEYS):
        index[status_key] = torch.full((8,), float(status_idx))
    index_path = tmp_path / "transfuser_feature.gz"
    _dump_pickle(index_path, index)

    features = load_legacy_features(index_path, tmp_path)

    assert features.images.shape == (4, 3, 8, 12, 3)
    assert features.geometry.shape == (4, 3, 9, 2048)
    assert features.ego_status.shape == (4, 8)


def test_build_structural_fixture_from_one_frame(tmp_path: Path) -> None:
    token = "frame"
    _write_frame(tmp_path, token)
    index = {key: torch.zeros(8) for key in LEGACY_STATUS_KEYS}
    index_path = tmp_path / "transfuser_feature.gz"
    _dump_pickle(index_path, index)

    features = build_structural_fixture(
        index_path=index_path,
        frame_cache_path=tmp_path / token / f"{token}.gz",
        geometry_path=tmp_path / token / "geometry_feature.npy",
    )

    assert features.images.shape[0] == 4
    assert torch.equal(features.images[0], features.images[-1])


def test_public_cache_dataset(tmp_path: Path) -> None:
    sample_root = tmp_path / "log" / "sample"
    _dump_pickle(
        sample_root / "latent_wam_features_v2.gz",
        {
            "images": torch.zeros((4, 3, 8, 12, 3), dtype=torch.uint8),
            "geometry": torch.zeros((4, 3, 9, 2048)),
            "ego_status": torch.zeros((4, 8)),
            "schema_version": 2,
        },
    )
    _dump_pickle(
        sample_root / "latent_wam_targets_v2.gz",
        {"trajectory": torch.zeros((8, 3))},
    )

    dataset = LatentWAMCacheDataset(tmp_path)
    features, targets = dataset[0]

    assert len(dataset) == 1
    assert features["images"].shape == (4, 3, 8, 12, 3)
    assert targets["trajectory"].shape == (8, 3)


def test_public_v1_cache_is_not_discovered(tmp_path: Path) -> None:
    sample_root = tmp_path / "log" / "sample"
    _dump_pickle(sample_root / "latent_wam_features_v1.gz", {})
    _dump_pickle(sample_root / "latent_wam_targets_v1.gz", {})

    assert len(LatentWAMCacheDataset(tmp_path)) == 0


def test_public_v2_cache_requires_schema_version(tmp_path: Path) -> None:
    sample_root = tmp_path / "log" / "sample"
    _dump_pickle(
        sample_root / "latent_wam_features_v2.gz",
        {
            "images": torch.zeros((4, 3, 8, 12, 3), dtype=torch.uint8),
            "geometry": torch.zeros((4, 3, 9, 2048)),
            "ego_status": torch.zeros((4, 8)),
        },
    )
    _dump_pickle(sample_root / "latent_wam_targets_v2.gz", {"trajectory": torch.zeros((8, 3))})

    with pytest.raises(ValueError, match="schema_version"):
        LatentWAMCacheDataset(tmp_path)[0]


def test_public_cache_is_atomically_published_and_validated(tmp_path: Path) -> None:
    features = {
        "images": torch.zeros((4, 3, 8, 12, 3), dtype=torch.uint8),
        "geometry": torch.zeros((4, 3, 9, 2048)),
        "ego_status": torch.zeros((4, 8)),
        "schema_version": 2,
    }
    targets = {"trajectory": torch.zeros((8, 3))}

    publish_public_cache(tmp_path, features, targets, rank=3, expected_num_poses=8)

    feature_path = tmp_path / PUBLIC_FEATURE_NAME
    target_path = tmp_path / PUBLIC_TARGET_NAME
    assert is_valid_public_cache(feature_path, target_path, expected_num_poses=8)
    assert not list(tmp_path.glob("*.tmp"))

    target_path.write_bytes(b"broken")
    assert not is_valid_public_cache(feature_path, target_path, expected_num_poses=8)
