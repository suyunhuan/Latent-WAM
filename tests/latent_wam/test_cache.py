from pathlib import Path

import pytest
import torch

import navsim.agents.latent_wam.dataset as dataset_module
from navsim.agents.latent_wam.cache import (
    CACHE_FORMAT,
    FrameRecord,
    SampleRecord,
    load_cache_manifest,
    load_cache_shard_record,
    publish_cache_index,
    stage_cache,
    write_cache_shard,
)
from navsim.agents.latent_wam.config import LatentWAMConfig, VisionConfig
from navsim.agents.latent_wam.dataset import LatentWAMCacheDataset


def _cache_tensors(num_frames: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images = torch.arange(num_frames, dtype=torch.uint8).view(num_frames, 1, 1, 1, 1).expand(num_frames, 3, 224, 448, 3)
    geometry = torch.arange(num_frames, dtype=torch.float32).view(num_frames, 1, 1, 1).expand(num_frames, 3, 519, 2048)
    ego_status = torch.arange(num_frames * 8, dtype=torch.float32).reshape(num_frames, 8)
    return images.contiguous(), geometry.contiguous(), ego_status


def _publish_fixture(cache_root: Path) -> None:
    frame_tokens = ("frame-0", "frame-1", "frame-2", "frame-3")
    images, geometry, ego_status = _cache_tensors(len(frame_tokens))
    fingerprint = {
        "dataset": {"name": "fixture"},
        "time_offsets_s": [-1.5, 0.0, 2.0, 4.0],
        "camera_names": ["cam_l0", "cam_f0", "cam_r0"],
        "preprocessing": {
            "revision": 1,
            "resize_size": [256, 455],
            "crop_size": [224, 448],
            "interpolation": "torch_bicubic_antialias",
        },
        "worldmirror": {
            "model_id": "tencent/HunyuanWorld-Mirror",
            "patch_start_index": 7,
        },
        "geometry": {"shape_per_frame": [3, 519, 2048], "dtype": "float32"},
    }
    shard = write_cache_shard(
        cache_root,
        0,
        frame_tokens,
        images,
        geometry,
        ego_status,
    )
    frames = [
        FrameRecord(token=token, log_name="log", timestamp=index, shard_id=0, row_index=index)
        for index, token in enumerate(frame_tokens)
    ]
    samples = [
        SampleRecord(
            token="sample",
            log_name="log",
            frame_tokens=frame_tokens,
            trajectory=torch.arange(24, dtype=torch.float32).reshape(8, 3),
        )
    ]
    publish_cache_index(
        cache_root,
        frames=frames,
        samples=samples,
        shards=[shard],
        fingerprint=fingerprint,
        frames_per_shard=64,
    )


def test_sharded_cache_reconstructs_sample_from_frame_index(tmp_path: Path) -> None:
    _publish_fixture(tmp_path)

    dataset = LatentWAMCacheDataset(tmp_path)
    features, targets = dataset[0]

    assert len(dataset) == 1
    assert features["token"] == "sample"
    assert features["images"].shape == (4, 3, 224, 448, 3)
    assert features["geometry"].shape == (4, 3, 519, 2048)
    assert torch.equal(features["images"][:, 0, 0, 0, 0], torch.arange(4, dtype=torch.uint8))
    assert torch.equal(features["ego_status"], torch.arange(32, dtype=torch.float32).reshape(4, 8))
    assert torch.equal(targets["trajectory"], torch.arange(24, dtype=torch.float32).reshape(8, 3))

    manifest = load_cache_manifest(tmp_path)
    assert manifest["format"] == CACHE_FORMAT
    assert manifest["status"] == "complete"
    assert manifest["frame_count"] == 4
    assert manifest["sample_count"] == 1


def test_cache_is_unavailable_until_final_manifest_is_published(tmp_path: Path) -> None:
    images, geometry, ego_status = _cache_tensors(1)
    write_cache_shard(tmp_path, 0, ("frame",), images, geometry, ego_status)

    with pytest.raises(ValueError, match="manifest"):
        LatentWAMCacheDataset(tmp_path)


def test_resumable_shard_is_bound_to_its_ordered_frame_tokens(tmp_path: Path) -> None:
    images, geometry, ego_status = _cache_tensors(1)
    write_cache_shard(tmp_path, 0, ("frame",), images, geometry, ego_status)

    assert load_cache_shard_record(tmp_path, 0, ("frame",)) is not None
    assert load_cache_shard_record(tmp_path, 0, ("other",)) is None


def test_cache_rejects_incompatible_fingerprint(tmp_path: Path) -> None:
    _publish_fixture(tmp_path)

    with pytest.raises(ValueError, match="fingerprint"):
        load_cache_manifest(tmp_path, expected_fingerprint={"dataset": "different"})


def test_training_configuration_is_strictly_checked_against_cache_semantics(tmp_path: Path) -> None:
    _publish_fixture(tmp_path)

    assert len(LatentWAMCacheDataset(tmp_path, config=LatentWAMConfig())) == 1
    with pytest.raises(ValueError, match="crop_size"):
        LatentWAMCacheDataset(tmp_path, config=LatentWAMConfig(vision=VisionConfig(crop_width=224)))


def test_training_cache_can_skip_redundant_header_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _publish_fixture(tmp_path)

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("header validation should be skipped")

    monkeypatch.setattr(dataset_module, "validate_cache_headers", fail_if_called)

    assert len(LatentWAMCacheDataset(tmp_path, validate_headers=False)) == 1


def test_frame_tokens_must_be_globally_unique(tmp_path: Path) -> None:
    _publish_fixture(tmp_path)
    manifest = load_cache_manifest(tmp_path)

    with pytest.raises(ValueError, match="duplicate frame token"):
        publish_cache_index(
            tmp_path,
            frames=[
                FrameRecord("same", "log", 0, 0, 0),
                FrameRecord("same", "log", 1, 0, 1),
            ],
            samples=[],
            shards=manifest["shards"],
            fingerprint={"dataset": "fixture"},
            frames_per_shard=64,
        )


def test_complete_cache_records_explicit_metadata_and_can_be_staged(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "local-nvme"
    _publish_fixture(source)

    source_manifest = load_cache_manifest(source)
    records = [source_manifest["schema"], *source_manifest["indices"].values()]
    for shard in source_manifest["shards"]:
        records.extend((shard["images"], shard["geometry"]))
    assert all(set(record) == {"path", "size_bytes", "shape", "dtype"} for record in records)

    staged = stage_cache(source, destination)

    assert staged == destination
    load_cache_manifest(destination)
    assert LatentWAMCacheDataset(destination)[0][0]["token"] == "sample"
