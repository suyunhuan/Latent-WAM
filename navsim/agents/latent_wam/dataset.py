"""Training dataset backed by the frame-deduplicated Latent WAM cache."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import torch
from safetensors.torch import load_file

from navsim.agents.latent_wam.cache import (
    LatentWAMFeatures,
    load_cache_manifest,
    validate_cache_compatibility,
    validate_cache_headers,
)


class _ShardLRU:
    """Small per-process cache of memory-mapped safetensors shard tensors."""

    def __init__(self, cache_root: Path, shards: Mapping[int, Mapping[str, Any]], capacity: int) -> None:
        if capacity < 1:
            raise ValueError("shard cache capacity must be positive")
        self.cache_root = cache_root
        self.shards = shards
        self.capacity = capacity
        self._entries: OrderedDict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = OrderedDict()

    def get(self, shard_id: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if shard_id in self._entries:
            self._entries.move_to_end(shard_id)
            return self._entries[shard_id]
        if shard_id not in self.shards:
            raise ValueError(f"unknown cache shard: {shard_id}")
        image_path = self.cache_root / self.shards[shard_id]["images"]["path"]
        geometry_path = self.cache_root / self.shards[shard_id]["geometry"]["path"]
        tensors = (
            load_file(str(image_path), device="cpu"),
            load_file(str(geometry_path), device="cpu"),
        )
        self._entries[shard_id] = tensors
        self._entries.move_to_end(shard_id)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)
        return tensors


class LatentWAMCacheDataset(torch.utils.data.Dataset):
    """Reconstruct four-frame samples from frame shards and Parquet indices.

    Args:
        cache_path: Complete cache root.
        log_names: Optional log filter.
        expected_fingerprint: Optional exact cache semantic identity.
        shard_cache_capacity: Number of aligned shard pairs retained per worker.
        config: Optional Latent WAM configuration for strict compatibility.
        dataset_split: Optional expected NAVSIM split.
        allowed_logs: Optional complete set of train/validation logs.
        validate_headers: Whether to inspect every safetensors header during startup.
    """

    def __init__(
        self,
        cache_path: str | Path,
        log_names: Iterable[str] | None = None,
        expected_fingerprint: Mapping[str, Any] | None = None,
        shard_cache_capacity: int = 8,
        config: Any | None = None,
        dataset_split: str | None = None,
        allowed_logs: Iterable[str] | None = None,
        validate_headers: bool = True,
    ) -> None:
        import pyarrow.parquet as pq

        self.cache_path = Path(cache_path)
        self.manifest = load_cache_manifest(self.cache_path, expected_fingerprint=expected_fingerprint)
        if config is not None:
            validate_cache_compatibility(
                self.manifest,
                config,
                dataset_split=dataset_split,
                allowed_logs=allowed_logs,
            )
        if validate_headers:
            validate_cache_headers(self.cache_path, self.manifest)
        self.shards = {int(shard["id"]): shard for shard in self.manifest["shards"]}
        frame_table = pq.read_table(self.cache_path / self.manifest["indices"]["frames"]["path"])
        sample_table = pq.read_table(self.cache_path / self.manifest["indices"]["samples"]["path"])
        self.frames = {row["token"]: (int(row["shard_id"]), int(row["row_index"])) for row in frame_table.to_pylist()}
        allowed_logs = set(log_names) if log_names is not None else None
        self.samples = [
            row for row in sample_table.to_pylist() if allowed_logs is None or row["log_name"] in allowed_logs
        ]
        self.sample_locality = [
            min(self.frames[token][0] for token in sample["frame_tokens"]) for sample in self.samples
        ]
        self.shard_cache_capacity = shard_cache_capacity
        self._shard_cache = _ShardLRU(self.cache_path, self.shards, shard_cache_capacity)

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        state["_shard_cache"] = None
        return state

    def _cache(self) -> _ShardLRU:
        if self._shard_cache is None:
            self._shard_cache = _ShardLRU(self.cache_path, self.shards, self.shard_cache_capacity)
        return self._shard_cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Dict[str, object], Dict[str, torch.Tensor]]:
        sample = self.samples[index]
        images = []
        geometry = []
        ego_status = []
        for frame_token in sample["frame_tokens"]:
            try:
                shard_id, row_index = self.frames[frame_token]
            except KeyError as error:
                raise ValueError(f"sample references an unknown frame token: {frame_token}") from error
            image_tensors, geometry_tensors = self._cache().get(shard_id)
            images.append(image_tensors["images"][row_index])
            geometry.append(geometry_tensors["geometry"][row_index])
            ego_status.append(geometry_tensors["ego_status"][row_index])
        features = LatentWAMFeatures(
            images=torch.stack(images),
            geometry=torch.stack(geometry),
            ego_status=torch.stack(ego_status),
        )
        features.validate()
        trajectory = torch.tensor(sample["trajectory"], dtype=torch.float32).reshape(8, 3)
        return (
            {
                "images": features.images,
                "geometry": features.geometry,
                "ego_status": features.ego_status,
                "token": sample["token"],
            },
            {"trajectory": trajectory},
        )
