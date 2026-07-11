"""Distributed coordination primitives for Latent WAM cache generation."""

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CacheFailure:
    """Failure details for one cache token."""

    token: str
    log_name: str
    error: str


@dataclass(frozen=True)
class RankCacheSummary:
    """Cache outcome counts produced by one rank."""

    rank: int
    assigned: int
    succeeded: int = 0
    skipped: int = 0
    failures: Tuple[CacheFailure, ...] = ()


@dataclass(frozen=True)
class DistributedCacheContext:
    """Rank and backend configuration for one cache process."""

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
        """Build a context from torchrun-compatible environment variables.

        Args:
            environment: Optional environment mapping used instead of ``os.environ``.
            cuda_available: Optional CUDA availability override, primarily for tests.

        Returns:
            Parsed rank, world-size, and process-group backend configuration.

        Raises:
            ValueError: If the rank-related environment values are inconsistent.
        """
        values = os.environ if environment is None else environment
        if cuda_available is None:
            import torch

            cuda_available = torch.cuda.is_available()
        world_size = int(values.get("WORLD_SIZE", "1"))
        rank = int(values.get("RANK", "0"))
        local_rank = int(values.get("LOCAL_RANK", "0"))
        backend = values.get("LATENT_WAM_DIST_BACKEND", "nccl" if cuda_available else "gloo")
        if world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("RANK must be in [0, WORLD_SIZE)")
        if local_rank < 0:
            raise ValueError("LOCAL_RANK must be non-negative")
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            backend=backend,
            cuda_available=cuda_available,
        )

    @property
    def is_distributed(self) -> bool:
        """Whether this process belongs to a multi-process cache job."""
        return self.world_size > 1

    @contextmanager
    def activate(self) -> Iterator[None]:
        """Initialize and finally destroy a process group owned by this context.

        Yields:
            Control after any required process group has been initialized.

        Raises:
            ValueError: If an existing process group does not match this context.
        """
        if not self.is_distributed:
            yield
            return
        import torch
        import torch.distributed as dist

        owns_process_group = not dist.is_initialized()
        if owns_process_group:
            if self.cuda_available:
                torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend=self.backend,
                rank=self.rank,
                world_size=self.world_size,
            )
        elif dist.get_rank() != self.rank or dist.get_world_size() != self.world_size:
            raise ValueError("existing process group does not match the cache environment")
        try:
            yield
        finally:
            if owns_process_group and dist.is_initialized():
                dist.destroy_process_group()

    def verify_token_set(self, tokens: Iterable[str]) -> None:
        """Fail on every rank unless all processes see the same token set.

        Args:
            tokens: Scene tokens visible to this rank.

        Raises:
            ValueError: If ranks see different token counts or fingerprints.
        """
        canonical = canonical_tokens(tokens)
        local_signature = (len(canonical), token_set_fingerprint(canonical))
        signatures = self.all_gather_object(local_signature)
        if any(signature != local_signature for signature in signatures):
            raise ValueError(f"cache token sets differ across ranks: {signatures}")

    def all_gather_object(self, value: Any) -> List[Any]:
        """Gather a Python value from every rank, or wrap it for one process.

        Args:
            value: Pickle-compatible value produced by the current rank.

        Returns:
            Values ordered by rank.
        """
        if not self.is_distributed:
            return [value]
        import torch.distributed as dist

        gathered: List[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def barrier(self) -> None:
        """Synchronize distributed cache ranks when a process group is active."""
        if self.is_distributed:
            import torch.distributed as dist

            dist.barrier()


def canonical_tokens(tokens: Iterable[str]) -> List[str]:
    """Return a sorted token list and reject duplicate identifiers.

    Args:
        tokens: Scene token identifiers.

    Returns:
        Lexicographically sorted identifiers.

    Raises:
        ValueError: If an identifier occurs more than once.
    """
    canonical = sorted(tokens)
    if len(canonical) != len(set(canonical)):
        raise ValueError("cache token list contains duplicate identifiers")
    return canonical


def token_set_fingerprint(tokens: Iterable[str]) -> str:
    """Compute an order-independent SHA-256 fingerprint for cache tokens.

    Args:
        tokens: Scene token identifiers.

    Returns:
        Hexadecimal SHA-256 digest of the canonical token list.
    """
    payload = json.dumps(canonical_tokens(tokens), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def plan_token_shard(tokens: Iterable[str], world_size: int, rank: int) -> List[str]:
    """Return the contiguous balanced token shard assigned to one rank.

    Args:
        tokens: Complete scene token collection.
        world_size: Number of cache workers.
        rank: Worker rank in ``[0, world_size)``.

    Returns:
        This rank's deterministic contiguous shard.

    Raises:
        ValueError: If ``world_size`` or ``rank`` is invalid.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if rank < 0 or rank >= world_size:
        raise ValueError("rank must be in [0, world_size)")
    canonical = canonical_tokens(tokens)
    shard_size, remainder = divmod(len(canonical), world_size)
    begin = rank * shard_size + min(rank, remainder)
    end = begin + shard_size + int(rank < remainder)
    return canonical[begin:end]


def build_cache_manifest(
    summaries: Sequence[RankCacheSummary],
    total_tokens: int,
    world_size: int,
    dry_run: bool,
) -> Dict[str, Any]:
    """Validate rank summaries and build the final cache manifest.

    Args:
        summaries: One complete outcome summary per rank.
        total_tokens: Number of tokens in the verified global token set.
        world_size: Number of cache workers.
        dry_run: Whether workers only verified distributed coordination.

    Returns:
        JSON-serializable aggregate and per-rank counts.

    Raises:
        ValueError: If summaries are incomplete or do not account for all tokens.
    """
    if len(summaries) != world_size or {summary.rank for summary in summaries} != set(range(world_size)):
        raise ValueError("cache summaries must contain every rank exactly once")
    assigned = sum(summary.assigned for summary in summaries)
    if assigned != total_tokens:
        raise ValueError("cache summaries do not cover the complete token set")
    if not dry_run:
        for summary in summaries:
            processed = summary.succeeded + summary.skipped + len(summary.failures)
            if processed != summary.assigned:
                raise ValueError(f"rank {summary.rank} did not account for every assigned token")
    failures = [asdict(failure) for summary in summaries for failure in summary.failures]
    succeeded = sum(summary.succeeded for summary in summaries)
    skipped = sum(summary.skipped for summary in summaries)
    status = "dry_run" if dry_run else ("failed" if failures else "complete")
    return {
        "status": status,
        "dry_run": dry_run,
        "world_size": world_size,
        "total_tokens": total_tokens,
        "assigned": assigned,
        "succeeded": succeeded,
        "skipped": skipped,
        "failed": len(failures),
        "failures": failures,
        "ranks": [
            {
                "rank": summary.rank,
                "assigned": summary.assigned,
                "succeeded": summary.succeeded,
                "skipped": summary.skipped,
                "failed": len(summary.failures),
            }
            for summary in sorted(summaries, key=lambda item: item.rank)
        ],
    }
