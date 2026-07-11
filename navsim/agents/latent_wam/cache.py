"""Frame-deduplicated sharded cache format for Latent WAM."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CACHE_FORMAT = "latent_wam_sharded_cache"
SCHEMA_REVISION = 1
TIME_OFFSETS_S: Tuple[float, ...] = (-1.5, 0.0, 2.0, 4.0)
MANIFEST_NAME = "manifest.json"
SCHEMA_NAME = "schema.json"
FRAMES_INDEX_PATH = "meta/frames.parquet"
SAMPLES_INDEX_PATH = "meta/samples.parquet"


@dataclass(frozen=True)
class LatentWAMFeatures:
    """One reconstructed training sample before DataLoader collation."""

    images: torch.Tensor
    geometry: torch.Tensor
    ego_status: torch.Tensor
    time_offsets_s: Tuple[float, ...] = TIME_OFFSETS_S

    def validate(self) -> None:
        """Validate sample alignment, dtype, and finite numeric values."""
        if self.images.ndim != 5 or self.images.shape[:2] != (4, 3) or self.images.shape[-1] != 3:
            raise ValueError("images must have shape [4,3,H,W,3]")
        if self.images.dtype != torch.uint8:
            raise ValueError("images must use uint8")
        if self.geometry.ndim != 4 or self.geometry.shape[:2] != (4, 3):
            raise ValueError("geometry must have shape [4,3,N,D]")
        if self.geometry.dtype != torch.float32:
            raise ValueError("geometry must use float32")
        if self.ego_status.shape != (4, 8) or self.ego_status.dtype != torch.float32:
            raise ValueError("ego_status must have shape [4,8] and use float32")
        if not torch.isfinite(self.geometry).all() or not torch.isfinite(self.ego_status).all():
            raise ValueError("cached numeric features must be finite")


@dataclass(frozen=True)
class FrameRecord:
    """Location of one globally unique NAVSIM frame in an immutable shard."""

    token: str
    log_name: str
    timestamp: int
    shard_id: int
    row_index: int


@dataclass(frozen=True)
class SampleRecord:
    """Four-frame training view and its future trajectory target."""

    token: str
    log_name: str
    frame_tokens: Tuple[str, str, str, str]
    trajectory: torch.Tensor


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file SHA-256 without loading it all into memory.

    Args:
        path: File to hash.
        chunk_size: Streaming read size in bytes.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_digest(fingerprint: Mapping[str, Any]) -> str:
    """Hash cache semantic inputs using canonical JSON.

    Args:
        fingerprint: JSON-serializable semantic input mapping.

    Returns:
        Lowercase hexadecimal SHA-256 digest.
    """
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def frame_tokens_digest(frame_tokens: Iterable[str]) -> str:
    """Hash the ordered frame rows in one shard.

    Args:
        frame_tokens: Frame tokens in physical row order.

    Returns:
        Order-sensitive hexadecimal SHA-256 digest.
    """
    payload = json.dumps(list(frame_tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_record(cache_root: Path, path: Path, shape: Sequence[int], dtype: torch.dtype) -> Dict[str, Any]:
    return {
        "path": path.relative_to(cache_root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "shape": list(shape),
        "dtype": str(dtype).removeprefix("torch."),
    }


def _validate_shard_tensors(
    frame_tokens: Sequence[str],
    images: torch.Tensor,
    geometry: torch.Tensor,
    ego_status: torch.Tensor,
) -> None:
    frame_count = len(frame_tokens)
    if frame_count < 1 or len(set(frame_tokens)) != frame_count or any(not token for token in frame_tokens):
        raise ValueError("shard frame tokens must be non-empty and unique")
    if images.shape != (frame_count, 3, 224, 448, 3):
        raise ValueError("shard images must have shape [N,3,224,448,3]")
    if images.dtype != torch.uint8:
        raise ValueError("shard images must use uint8")
    if geometry.shape != (frame_count, 3, 519, 2048):
        raise ValueError("shard geometry must have shape [N,3,519,2048]")
    if geometry.dtype != torch.float32:
        raise ValueError("shard geometry must use float32")
    if ego_status.shape != (frame_count, 8) or ego_status.dtype != torch.float32:
        raise ValueError("shard ego_status must have shape [N,8] and use float32")
    if not torch.isfinite(geometry).all() or not torch.isfinite(ego_status).all():
        raise ValueError("shard tensors must contain finite values")


def _save_safetensors_atomic(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.pid-{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        save_file({key: value.detach().cpu().contiguous() for key, value in tensors.items()}, str(temporary_path))
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_cache_shard(
    cache_root: Path,
    shard_id: int,
    frame_tokens: Sequence[str],
    images: torch.Tensor,
    geometry: torch.Tensor,
    ego_status: torch.Tensor,
    cache_fingerprint_sha256: str,
) -> Dict[str, Any]:
    """Validate and atomically publish one aligned shard pair.

    Args:
        cache_root: Cache root directory.
        shard_id: Non-negative physical shard identifier.
        frame_tokens: Unique frame tokens in tensor row order.
        images: Preprocessed uint8 image tensor.
        geometry: Float32 WorldMirror tensor.
        ego_status: Float32 frame status tensor.
        cache_fingerprint_sha256: Semantic identity binding for resume.

    Returns:
        JSON-serializable shard manifest record.

    Raises:
        ValueError: If IDs, fingerprints, shapes, dtypes, or values are invalid.
    """
    cache_root = Path(cache_root)
    if shard_id < 0:
        raise ValueError("shard_id must be non-negative")
    if len(cache_fingerprint_sha256) != 64:
        raise ValueError("cache fingerprint must be a SHA-256 hexadecimal digest")
    _validate_shard_tensors(frame_tokens, images, geometry, ego_status)
    image_path = cache_root / "images" / f"shard-{shard_id:06d}.safetensors"
    geometry_path = cache_root / "geometry" / f"shard-{shard_id:06d}.safetensors"
    _save_safetensors_atomic(image_path, {"images": images})
    _save_safetensors_atomic(geometry_path, {"geometry": geometry, "ego_status": ego_status})
    record = {
        "id": shard_id,
        "frame_count": len(frame_tokens),
        "frame_tokens_sha256": frame_tokens_digest(frame_tokens),
        "cache_fingerprint_sha256": cache_fingerprint_sha256,
        "first_frame_token": frame_tokens[0],
        "last_frame_token": frame_tokens[-1],
        "images": _file_record(cache_root, image_path, images.shape, images.dtype),
        "geometry": _file_record(cache_root, geometry_path, geometry.shape, geometry.dtype),
        "ego_status": {"shape": list(ego_status.shape), "dtype": "float32"},
    }
    sidecar_path = cache_root / "meta" / "shards" / f"shard-{shard_id:06d}.json"
    _atomic_write_text(sidecar_path, json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def load_cache_shard_record(
    cache_root: Path,
    shard_id: int,
    expected_frame_tokens: Sequence[str],
    expected_cache_fingerprint_sha256: str,
    verify_files: bool = True,
) -> Dict[str, Any] | None:
    """Load and validate a resumable shard sidecar.

    Args:
        cache_root: Cache root directory.
        shard_id: Expected shard identifier.
        expected_frame_tokens: Expected row-ordered frame tokens.
        expected_cache_fingerprint_sha256: Expected semantic cache identity.
        verify_files: Whether to recompute full file checksums.

    Returns:
        Shard record, or ``None`` when the shard must be rebuilt.
    """
    cache_root = Path(cache_root)
    sidecar_path = cache_root / "meta" / "shards" / f"shard-{shard_id:06d}.json"
    if not sidecar_path.is_file():
        return None
    try:
        record = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        record.get("id") != shard_id
        or record.get("frame_count") != len(expected_frame_tokens)
        or record.get("frame_tokens_sha256") != frame_tokens_digest(expected_frame_tokens)
        or record.get("cache_fingerprint_sha256") != expected_cache_fingerprint_sha256
        or record.get("images", {}).get("shape") != [len(expected_frame_tokens), 3, 224, 448, 3]
        or record.get("geometry", {}).get("shape") != [len(expected_frame_tokens), 3, 519, 2048]
        or record.get("ego_status") != {"shape": [len(expected_frame_tokens), 8], "dtype": "float32"}
    ):
        return None
    for kind in ("images", "geometry"):
        file_record = record.get(kind, {})
        path = cache_root / file_record.get("path", "missing")
        if not path.is_file() or path.stat().st_size != file_record.get("size_bytes"):
            return None
        if verify_files and sha256_file(path) != file_record.get("sha256"):
            return None
    return record


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.pid-{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_index_records(
    frames: Sequence[FrameRecord], samples: Sequence[SampleRecord], shards: Sequence[Mapping[str, Any]]
) -> None:
    frame_tokens = [frame.token for frame in frames]
    if len(frame_tokens) != len(set(frame_tokens)):
        raise ValueError("duplicate frame token in cache index")
    shard_counts = {int(shard["id"]): int(shard["frame_count"]) for shard in shards}
    if len(shard_counts) != len(shards):
        raise ValueError("duplicate shard id in cache manifest")
    occupied_rows = set()
    for frame in frames:
        if frame.shard_id not in shard_counts or not 0 <= frame.row_index < shard_counts[frame.shard_id]:
            raise ValueError(f"frame {frame.token} references an invalid shard row")
        location = (frame.shard_id, frame.row_index)
        if location in occupied_rows:
            raise ValueError("multiple frame tokens reference the same shard row")
        occupied_rows.add(location)
    if len(occupied_rows) != sum(shard_counts.values()):
        raise ValueError("frame index does not cover every shard row")
    known_frames = set(frame_tokens)
    sample_tokens = set()
    for sample in samples:
        if sample.token in sample_tokens:
            raise ValueError("duplicate sample token in cache index")
        sample_tokens.add(sample.token)
        if len(sample.frame_tokens) != 4 or not set(sample.frame_tokens).issubset(known_frames):
            raise ValueError(f"sample {sample.token} references unknown or malformed frames")
        trajectory = torch.as_tensor(sample.trajectory)
        if trajectory.shape != (8, 3) or trajectory.dtype != torch.float32 or not torch.isfinite(trajectory).all():
            raise ValueError("sample trajectories must have finite float32 shape [8,3]")


def _index_tables(frames: Sequence[FrameRecord], samples: Sequence[SampleRecord]) -> Tuple[Any, Any]:
    import pyarrow as pa

    frames_table = pa.table(
        {
            "token": [frame.token for frame in frames],
            "log_name": [frame.log_name for frame in frames],
            "timestamp": pa.array([frame.timestamp for frame in frames], type=pa.int64()),
            "shard_id": pa.array([frame.shard_id for frame in frames], type=pa.int32()),
            "row_index": pa.array([frame.row_index for frame in frames], type=pa.int32()),
        }
    )
    samples_table = pa.table(
        {
            "token": [sample.token for sample in samples],
            "log_name": [sample.log_name for sample in samples],
            "frame_tokens": [list(sample.frame_tokens) for sample in samples],
            "trajectory": [torch.as_tensor(sample.trajectory).reshape(-1).tolist() for sample in samples],
        }
    )
    return frames_table, samples_table


def _write_parquet_atomic(path: Path, table: Any) -> None:
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        pq.write_table(table, temporary_path, compression="zstd")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_parquet_indices(
    cache_root: Path, frames: Sequence[FrameRecord], samples: Sequence[SampleRecord]
) -> Tuple[Path, Path]:
    frames_path = cache_root / FRAMES_INDEX_PATH
    samples_path = cache_root / SAMPLES_INDEX_PATH
    frames_table, samples_table = _index_tables(frames, samples)
    _write_parquet_atomic(frames_path, frames_table)
    _write_parquet_atomic(samples_path, samples_table)
    return frames_path, samples_path


def write_partial_cache_index(
    cache_root: Path,
    rank: int,
    frames: Sequence[FrameRecord],
    samples: Sequence[SampleRecord],
) -> Tuple[Path, Path]:
    """Write one rank's Parquet index fragments atomically.

    Args:
        cache_root: Cache root directory.
        rank: Non-negative distributed rank.
        frames: Frame rows owned by the rank.
        samples: Sample rows assigned to the rank.

    Returns:
        Frame-fragment and sample-fragment paths.
    """
    if rank < 0:
        raise ValueError("rank must be non-negative")
    partial_root = Path(cache_root) / "meta" / "partial"
    partial_root.mkdir(parents=True, exist_ok=True)
    frames_path = partial_root / f"rank-{rank:05d}.frames.parquet"
    samples_path = partial_root / f"rank-{rank:05d}.samples.parquet"
    frames_table, samples_table = _index_tables(frames, samples)
    for path, table in ((frames_path, frames_table), (samples_path, samples_table)):
        _write_parquet_atomic(path, table)
    return frames_path, samples_path


def read_partial_cache_indices(cache_root: Path, world_size: int) -> Tuple[list[FrameRecord], list[SampleRecord]]:
    """Read every rank fragment for final validation and merge.

    Args:
        cache_root: Cache root directory.
        world_size: Number of expected rank fragments.

    Returns:
        Merged frame and sample records.

    Raises:
        ValueError: If an expected fragment is missing.
    """
    import pyarrow.parquet as pq

    frames: list[FrameRecord] = []
    samples: list[SampleRecord] = []
    partial_root = Path(cache_root) / "meta" / "partial"
    for rank in range(world_size):
        frames_path = partial_root / f"rank-{rank:05d}.frames.parquet"
        samples_path = partial_root / f"rank-{rank:05d}.samples.parquet"
        if not frames_path.is_file() or not samples_path.is_file():
            raise ValueError(f"cache index fragment is missing for rank {rank}")
        frames.extend(FrameRecord(**row) for row in pq.read_table(frames_path).to_pylist())
        samples.extend(
            SampleRecord(
                token=row["token"],
                log_name=row["log_name"],
                frame_tokens=tuple(row["frame_tokens"]),
                trajectory=torch.tensor(row["trajectory"], dtype=torch.float32).reshape(8, 3),
            )
            for row in pq.read_table(samples_path).to_pylist()
        )
    return frames, samples


def publish_cache_index(
    cache_root: Path,
    frames: Sequence[FrameRecord],
    samples: Sequence[SampleRecord],
    shards: Sequence[Mapping[str, Any]],
    fingerprint: Mapping[str, Any],
    frames_per_shard: int,
) -> Dict[str, Any]:
    """Publish indices and make a complete cache visible.

    Args:
        cache_root: Cache root directory.
        frames: Complete globally unique frame index.
        samples: Complete sample index.
        shards: Validated immutable shard records.
        fingerprint: Cache semantic identity mapping.
        frames_per_shard: Configured maximum frames per shard.

    Returns:
        Published manifest mapping.

    Raises:
        ValueError: If records are incomplete, duplicated, or inconsistent.
    """
    cache_root = Path(cache_root)
    if frames_per_shard < 1:
        raise ValueError("frames_per_shard must be positive")
    ordered_frames = sorted(frames, key=lambda frame: (frame.shard_id, frame.row_index))
    ordered_shards = sorted((dict(shard) for shard in shards), key=lambda shard: int(shard["id"]))
    _validate_index_records(ordered_frames, samples, ordered_shards)
    frames_path, samples_path = _write_parquet_indices(cache_root, ordered_frames, samples)
    schema = {
        "format": CACHE_FORMAT,
        "schema_revision": SCHEMA_REVISION,
        "time_offsets_s": list(TIME_OFFSETS_S),
        "frames_index": FRAMES_INDEX_PATH,
        "samples_index": SAMPLES_INDEX_PATH,
        "image_layout": "NVHWC",
        "geometry_layout": "NVPD",
    }
    schema_path = cache_root / SCHEMA_NAME
    _atomic_write_text(schema_path, json.dumps(schema, indent=2, sort_keys=True) + "\n")
    semantic_inputs = json.loads(json.dumps(fingerprint, sort_keys=True))
    manifest = {
        "format": CACHE_FORMAT,
        "schema_revision": SCHEMA_REVISION,
        "status": "complete",
        "frames_per_shard": frames_per_shard,
        "frame_count": len(ordered_frames),
        "sample_count": len(samples),
        "fingerprint": semantic_inputs,
        "fingerprint_sha256": fingerprint_digest(semantic_inputs),
        "schema": _file_record(cache_root, schema_path, [1], torch.uint8),
        "indices": {
            "frames": _file_record(cache_root, frames_path, [len(ordered_frames)], torch.uint8),
            "samples": _file_record(cache_root, samples_path, [len(samples)], torch.uint8),
        },
        "shards": ordered_shards,
    }
    _atomic_write_text(cache_root / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_cache_manifest(
    cache_root: Path,
    expected_fingerprint: Mapping[str, Any] | None = None,
    verify_files: bool = False,
    verify_presence: bool = True,
) -> Dict[str, Any]:
    """Load and validate a complete cache manifest.

    Args:
        cache_root: Cache root directory.
        expected_fingerprint: Optional exact semantic identity.
        verify_files: Whether to recompute every recorded checksum.
        verify_presence: Whether to check recorded paths and sizes.

    Returns:
        Validated manifest mapping.

    Raises:
        ValueError: If the manifest, identity, or files are invalid.
    """
    cache_root = Path(cache_root)
    manifest_path = cache_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"cache manifest is missing: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cache manifest is invalid: {manifest_path}") from error
    if (
        manifest.get("format") != CACHE_FORMAT
        or manifest.get("schema_revision") != SCHEMA_REVISION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("cache manifest format, revision, or status is incompatible")
    actual_fingerprint = manifest.get("fingerprint")
    if manifest.get("fingerprint_sha256") != fingerprint_digest(actual_fingerprint):
        raise ValueError("cache fingerprint digest is invalid")
    if expected_fingerprint is not None and fingerprint_digest(expected_fingerprint) != manifest["fingerprint_sha256"]:
        raise ValueError("cache fingerprint is incompatible with the requested training configuration")
    if verify_presence:
        records = [manifest["schema"], *manifest["indices"].values()]
        for shard in manifest["shards"]:
            records.extend((shard["images"], shard["geometry"]))
        for record in records:
            path = cache_root / record["path"]
            if not path.is_file() or path.stat().st_size != record["size_bytes"]:
                raise ValueError(f"cache file is missing or has the wrong size: {path}")
            if verify_files and sha256_file(path) != record["sha256"]:
                raise ValueError(f"cache file checksum mismatch: {path}")
    return manifest


def validate_cache_compatibility(
    manifest: Mapping[str, Any],
    config: Any,
    dataset_split: str | None = None,
    allowed_logs: Iterable[str] | None = None,
    worldmirror_weights_sha256: str | None = None,
) -> None:
    """Compare cache semantics with a training configuration.

    Args:
        manifest: Validated cache manifest.
        config: Latent WAM model configuration.
        dataset_split: Optional expected NAVSIM data split.
        allowed_logs: Optional complete set of train/validation log names.
        worldmirror_weights_sha256: Optional expected WorldMirror weights digest.

    Raises:
        ValueError: If any requested semantic field is incompatible.
    """
    fingerprint = manifest["fingerprint"]
    expected = {
        "time_offsets_s": list(config.data.time_offsets_s),
        "camera_names": ["cam_l0", "cam_f0", "cam_r0"],
        "crop_size": [config.vision.crop_height, config.vision.crop_width],
        "patch_start_index": config.geometry.patch_start_index,
        "geometry_shape": [config.data.num_views, 519, config.geometry.feature_dim],
        "geometry_dtype": "float32",
        "preprocessing_revision": 1,
        "resize_size": [256, 455],
        "interpolation": "torch_bicubic_antialias",
        "worldmirror_model_id": str(config.geometry.model_id),
        "worldmirror_weights_sha256": worldmirror_weights_sha256,
    }
    actual = {
        "time_offsets_s": fingerprint.get("time_offsets_s"),
        "camera_names": fingerprint.get("camera_names"),
        "crop_size": fingerprint.get("preprocessing", {}).get("crop_size"),
        "patch_start_index": fingerprint.get("worldmirror", {}).get("patch_start_index"),
        "geometry_shape": fingerprint.get("geometry", {}).get("shape_per_frame"),
        "geometry_dtype": fingerprint.get("geometry", {}).get("dtype"),
        "preprocessing_revision": fingerprint.get("preprocessing", {}).get("revision"),
        "resize_size": fingerprint.get("preprocessing", {}).get("resize_size"),
        "interpolation": fingerprint.get("preprocessing", {}).get("interpolation"),
        "worldmirror_model_id": fingerprint.get("worldmirror", {}).get("model_id"),
        "worldmirror_weights_sha256": fingerprint.get("worldmirror", {}).get("weights_sha256"),
    }
    if worldmirror_weights_sha256 is None:
        expected.pop("worldmirror_weights_sha256")
        actual.pop("worldmirror_weights_sha256")
    if dataset_split is not None:
        expected["dataset"] = {"name": "NAVSIM", "version": "2.1.2", "split": dataset_split}
        actual["dataset"] = fingerprint.get("dataset")
    if actual != expected:
        mismatches = [key for key in expected if actual[key] != expected[key]]
        raise ValueError(f"cache is incompatible with the training configuration: {mismatches}")
    if allowed_logs is not None:
        unexpected_logs = sorted(set(fingerprint.get("logs", [])).difference(allowed_logs))
        if unexpected_logs:
            raise ValueError(
                f"cache contains logs outside the configured train/validation split: {unexpected_logs[:3]}"
            )


def validate_cache_headers(cache_root: Path, manifest: Mapping[str, Any]) -> None:
    """Validate safetensors headers without reading payloads.

    Args:
        cache_root: Cache root directory.
        manifest: Validated cache manifest.

    Raises:
        ValueError: If keys, shapes, or dtypes differ from the manifest.
    """
    cache_root = Path(cache_root)
    dtype_names = {"U8": "uint8", "F32": "float32"}
    for shard in manifest["shards"]:
        image_path = cache_root / shard["images"]["path"]
        with safe_open(image_path, framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != {"images"}:
                raise ValueError(f"image shard has incompatible tensor keys: {image_path}")
            image_slice = tensors.get_slice("images")
            image_dtype = dtype_names.get(image_slice.get_dtype(), image_slice.get_dtype())
            if list(image_slice.get_shape()) != shard["images"]["shape"] or image_dtype != shard["images"]["dtype"]:
                raise ValueError(f"image shard header is incompatible: {image_path}")
        geometry_path = cache_root / shard["geometry"]["path"]
        with safe_open(geometry_path, framework="pt", device="cpu") as tensors:
            if set(tensors.keys()) != {"geometry", "ego_status"}:
                raise ValueError(f"geometry shard has incompatible tensor keys: {geometry_path}")
            for key, record in (("geometry", shard["geometry"]), ("ego_status", shard["ego_status"])):
                tensor_slice = tensors.get_slice(key)
                dtype = dtype_names.get(tensor_slice.get_dtype(), tensor_slice.get_dtype())
                if list(tensor_slice.get_shape()) != record["shape"] or dtype != record["dtype"]:
                    raise ValueError(f"geometry shard header is incompatible for {key}: {geometry_path}")


def _manifest_file_records(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = [manifest["schema"], *manifest["indices"].values()]
    for shard in manifest["shards"]:
        records.extend((shard["images"], shard["geometry"]))
    return records


def stage_cache(source_root: Path, destination_root: Path) -> Path:
    """Copy a complete cache to node-local storage.

    Args:
        source_root: Complete shared cache root.
        destination_root: Node-local staging directory.

    Returns:
        Resolved node-local cache root.

    Raises:
        ValueError: If source or copied files fail validation.
    """
    import fcntl

    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    if source_root == destination_root:
        load_cache_manifest(source_root)
        return destination_root
    source_manifest = load_cache_manifest(source_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    lock_path = destination_root / ".stage.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        destination_manifest_path = destination_root / MANIFEST_NAME
        if destination_manifest_path.is_file():
            try:
                destination_manifest = load_cache_manifest(destination_root, verify_files=True)
                if destination_manifest["fingerprint_sha256"] == source_manifest["fingerprint_sha256"]:
                    return destination_root
            except ValueError:
                pass
            destination_manifest_path.unlink(missing_ok=True)
        for record in _manifest_file_records(source_manifest):
            source_path = source_root / record["path"]
            destination_path = destination_root / record["path"]
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = destination_path.with_name(f".{destination_path.name}.{uuid.uuid4().hex}.tmp")
            try:
                shutil.copyfile(source_path, temporary_path)
                wrong_size = temporary_path.stat().st_size != record["size_bytes"]
                wrong_checksum = sha256_file(temporary_path) != record["sha256"]
                if wrong_size or wrong_checksum:
                    raise ValueError(f"staged cache file failed checksum validation: {source_path}")
                os.replace(temporary_path, destination_path)
            finally:
                temporary_path.unlink(missing_ok=True)
        _atomic_write_text(
            destination_manifest_path,
            json.dumps(source_manifest, indent=2, sort_keys=True) + "\n",
        )
        load_cache_manifest(destination_root, verify_files=True)
    return destination_root
