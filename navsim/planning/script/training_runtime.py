"""Distributed runtime helpers for reproducible training jobs."""

import logging
import os
from pathlib import Path
from typing import MutableMapping, Optional

EXPERIMENT_UID_ENV = "NAVSIM_EXPERIMENT_UID"


def pin_experiment_uid(
    experiment_uid: str,
    environment: Optional[MutableMapping[str, str]] = None,
) -> str:
    """Pin an experiment identifier for child processes spawned by Lightning.

    Args:
        experiment_uid: Identifier resolved by the parent Hydra process.
        environment: Optional mutable environment mapping used instead of
            ``os.environ``.

    Returns:
        The inherited identifier when already pinned, otherwise
        ``experiment_uid``.
    """
    values = os.environ if environment is None else environment
    return values.setdefault(EXPERIMENT_UID_ENV, experiment_uid)


def distributed_rank(environment: Optional[MutableMapping[str, str]] = None) -> int:
    """Return the best available global rank before Lightning initialization.

    Args:
        environment: Optional environment mapping used instead of ``os.environ``.

    Returns:
        Global rank, falling back to the local rank and then zero.
    """
    values = os.environ if environment is None else environment
    if "RANK" in values:
        return int(values["RANK"])
    local_rank = int(values.get("LOCAL_RANK", "0"))
    node_rank = int(values.get("NODE_RANK", "0"))
    local_world_size = int(values.get("LOCAL_WORLD_SIZE", "1"))
    return node_rank * local_world_size + local_rank


def configure_rank_logging(
    output_dir: Path,
    rank: int,
    root_logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Keep Hydra logging on rank zero and isolate non-zero rank diagnostics.

    Args:
        output_dir: Shared experiment output directory.
        rank: Global distributed rank.
        root_logger: Optional logger override used by tests.

    Returns:
        Non-zero rank log path, or ``None`` for rank zero.
    """
    if rank == 0:
        return None
    logger = logging.getLogger() if root_logger is None else root_logger
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_log = output_dir / f"rank_{rank}.log"
    handler = logging.FileHandler(rank_log)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"))
    logger.addHandler(handler)
    return rank_log
