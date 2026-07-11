"""Global planning and distributed coordination for sharded cache generation."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class FrameSource:
    """Canonical sample location from which one unique frame is loaded."""

    sample_token: str
    frame_index: int


@dataclass(frozen=True)
class PlannedFrame:
    """Globally unique frame and its immutable shard row."""

    token: str
    log_name: str
    timestamp: int
    shard_id: int
    row_index: int


@dataclass(frozen=True)
class PlannedSample:
    """Training sample expressed as four references into the frame store."""

    token: str
    log_name: str
    frame_tokens: Tuple[str, str, str, str]


@dataclass(frozen=True)
class FrameCachePlan:
    """Deterministic global cache plan computed independently by every rank."""

    frames: Tuple[PlannedFrame, ...]
    samples: Tuple[PlannedSample, ...]
    sources: Mapping[str, FrameSource]
    frames_per_shard: int

    @property
    def num_shards(self) -> int:
        """Number of immutable frame shards in this plan."""
        return 0 if not self.frames else self.frames[-1].shard_id + 1

    def fingerprint(self) -> str:
        """Return a cross-rank comparable digest of all planned identities."""
        payload = {
            "frames": [frame.__dict__ for frame in self.frames],
            "samples": [sample.__dict__ for sample in self.samples],
            "sources": {token: source.__dict__ for token, source in sorted(self.sources.items())},
            "frames_per_shard": self.frames_per_shard,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _frame_semantic_digest(frame: Mapping[str, Any]) -> str:
    cameras = frame.get("cams", frame.get("camera_dict", {}))
    relevant = {
        "token": frame.get("token"),
        "timestamp": frame.get("timestamp"),
        "log_name": frame.get("log_name"),
        "cams": cameras,
        "ego2global_translation": frame.get("ego2global_translation"),
        "ego2global_rotation": frame.get("ego2global_rotation"),
        "ego_dynamic_state": frame.get("ego_dynamic_state"),
        "driving_command": frame.get("driving_command"),
    }
    payload = json.dumps(_json_value(relevant), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_frame_cache_plan(
    scenes: Mapping[str, Sequence[Mapping[str, Any]]],
    frame_indices: Sequence[int],
    frames_per_shard: int,
) -> FrameCachePlan:
    """Deduplicate selected frames and assign temporal shards.

    Args:
        scenes: Raw NAVSIM frame dictionaries keyed by sample token.
        frame_indices: Four required temporal indices.
        frames_per_shard: Maximum physical rows per shard.

    Returns:
        Deterministic global frame/sample/source plan.

    Raises:
        ValueError: If frames are missing, conflicting, or parameters are invalid.
    """
    if tuple(frame_indices) != (0, 3, 7, 11):
        raise ValueError("Latent WAM cache frames must use indices (0,3,7,11)")
    if frames_per_shard < 1:
        raise ValueError("frames_per_shard must be positive")
    sources: Dict[str, FrameSource] = {}
    frame_metadata: Dict[str, Tuple[str, int]] = {}
    frame_digests: Dict[str, str] = {}
    samples: List[PlannedSample] = []
    for sample_token in sorted(scenes):
        scene_frames = scenes[sample_token]
        if len(scene_frames) <= max(frame_indices):
            raise ValueError(f"sample {sample_token} does not contain all required frame indices")
        selected_tokens = []
        for frame_index in frame_indices:
            frame = scene_frames[frame_index]
            frame_token = str(frame["token"])
            log_name = str(frame["log_name"])
            timestamp = int(frame["timestamp"])
            digest = _frame_semantic_digest(frame)
            if frame_token in frame_digests and frame_digests[frame_token] != digest:
                raise ValueError(f"frame token {frame_token} has conflicting source data")
            frame_digests.setdefault(frame_token, digest)
            frame_metadata.setdefault(frame_token, (log_name, timestamp))
            candidate = FrameSource(sample_token=sample_token, frame_index=frame_index)
            previous = sources.get(frame_token)
            if previous is None or (candidate.sample_token, candidate.frame_index) < (
                previous.sample_token,
                previous.frame_index,
            ):
                sources[frame_token] = candidate
            selected_tokens.append(frame_token)
        samples.append(
            PlannedSample(
                token=sample_token,
                log_name=str(scene_frames[frame_indices[1]]["log_name"]),
                frame_tokens=tuple(selected_tokens),
            )
        )
    ordered_tokens = sorted(frame_metadata, key=lambda token: (*frame_metadata[token], token))
    frames = tuple(
        PlannedFrame(
            token=token,
            log_name=frame_metadata[token][0],
            timestamp=frame_metadata[token][1],
            shard_id=index // frames_per_shard,
            row_index=index % frames_per_shard,
        )
        for index, token in enumerate(ordered_tokens)
    )
    return FrameCachePlan(
        frames=frames,
        samples=tuple(samples),
        sources=sources,
        frames_per_shard=frames_per_shard,
    )


def plan_shard_ids(num_shards: int, world_size: int, rank: int) -> List[int]:
    """Assign immutable shards using round-robin ownership.

    Args:
        num_shards: Total number of shards.
        world_size: Number of owner ranks.
        rank: Current global rank.

    Returns:
        Shard IDs owned by the rank.

    Raises:
        ValueError: If counts or rank are invalid.
    """
    if num_shards < 0 or world_size < 1 or rank < 0 or rank >= world_size:
        raise ValueError("invalid shard ownership inputs")
    return list(range(rank, num_shards, world_size))


@dataclass(frozen=True)
class DistributedCacheContext:
    """Torchrun rank and process-group lifecycle for cache generation."""

    rank: int
    local_rank: int
    world_size: int
    backend: str
    cuda_available: bool

    @classmethod
    def from_environment(
        cls,
        environment: Optional[Mapping[str, str]] = None,
        cuda_available: Optional[bool] = None,
    ) -> "DistributedCacheContext":
        """Build a context from torchrun environment variables.

        Args:
            environment: Optional environment mapping for tests.
            cuda_available: Optional accelerator availability override.

        Returns:
            Parsed distributed cache context.

        Raises:
            ValueError: If rank variables are inconsistent.
        """
        values = os.environ if environment is None else environment
        if cuda_available is None:
            import torch

            cuda_available = torch.cuda.is_available()
        world_size = int(values.get("WORLD_SIZE", "1"))
        rank = int(values.get("RANK", "0"))
        local_rank = int(values.get("LOCAL_RANK", "0"))
        if world_size < 1 or rank < 0 or rank >= world_size or local_rank < 0:
            raise ValueError("invalid torchrun rank environment")
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=values.get("LATENT_WAM_DIST_BACKEND", "nccl" if cuda_available else "gloo"),
            cuda_available=cuda_available,
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Activate and, when owned, destroy the process group.

        Yields:
            Control while the matching process group is active.
        """
        if not self.is_distributed:
            if self.cuda_available:
                import torch

                torch.cuda.set_device(self.local_rank)
            yield
            return
        import torch
        import torch.distributed as dist

        owns_group = not dist.is_initialized()
        if owns_group:
            if self.cuda_available:
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(self.backend, rank=self.rank, world_size=self.world_size)
        try:
            yield
        finally:
            if owns_group and dist.is_initialized():
                dist.destroy_process_group()

    def all_gather_object(self, value: Any) -> List[Any]:
        """Gather a Python value from every rank.

        Args:
            value: Pickle-compatible local value.

        Returns:
            Values ordered by global rank.
        """
        if not self.is_distributed:
            return [value]
        import torch.distributed as dist

        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def verify_plan(self, plan: FrameCachePlan) -> None:
        """Require every rank to hold the same global plan.

        Args:
            plan: Locally computed cache plan.

        Raises:
            ValueError: If rank fingerprints differ.
        """
        signatures = self.all_gather_object(plan.fingerprint())
        if any(signature != signatures[0] for signature in signatures):
            raise ValueError(f"cache plans differ across ranks: {signatures}")

    def broadcast_object(self, value: Any, source_rank: int = 0) -> Any:
        """Broadcast one small Python object.

        Args:
            value: Source value, or a placeholder on other ranks.
            source_rank: Broadcasting rank.

        Returns:
            Broadcast value on every rank.
        """
        if not self.is_distributed:
            return value
        import torch.distributed as dist

        values = [value if self.rank == source_rank else None]
        dist.broadcast_object_list(values, src=source_rank)
        return values[0]

    def barrier(self) -> None:
        if self.is_distributed:
            import torch.distributed as dist

            dist.barrier()
