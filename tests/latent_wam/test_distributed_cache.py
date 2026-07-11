import os
import subprocess
import sys

import pytest

from navsim.agents.latent_wam.distributed_cache import (
    CacheFailure,
    DistributedCacheContext,
    RankCacheSummary,
    build_cache_manifest,
    plan_token_shard,
    token_set_fingerprint,
)


def test_token_shards_are_contiguous_and_balanced() -> None:
    tokens = ["echo", "alpha", "delta", "bravo", "charlie"]

    assert plan_token_shard(tokens, world_size=3, rank=0) == ["alpha", "bravo"]
    assert plan_token_shard(tokens, world_size=3, rank=1) == ["charlie", "delta"]
    assert plan_token_shard(tokens, world_size=3, rank=2) == ["echo"]


def test_token_shards_allow_more_ranks_than_tokens() -> None:
    shards = [plan_token_shard(["b", "a"], world_size=4, rank=rank) for rank in range(4)]

    assert shards == [["a"], ["b"], [], []]


def test_token_fingerprint_is_order_independent() -> None:
    assert token_set_fingerprint(["b", "a"]) == token_set_fingerprint(["a", "b"])


def test_distributed_context_reads_torchrun_environment() -> None:
    context = DistributedCacheContext.from_environment(
        {
            "RANK": "3",
            "LOCAL_RANK": "1",
            "WORLD_SIZE": "8",
            "LATENT_WAM_DIST_BACKEND": "gloo",
        },
        cuda_available=False,
    )

    assert context.rank == 3
    assert context.local_rank == 1
    assert context.world_size == 8
    assert context.backend == "gloo"
    assert context.is_distributed


def test_single_process_context_defaults_to_gloo_without_cuda() -> None:
    context = DistributedCacheContext.from_environment({}, cuda_available=False)

    assert context.rank == 0
    assert context.local_rank == 0
    assert context.world_size == 1
    assert context.backend == "gloo"
    assert not context.is_distributed


def test_cache_manifest_aggregates_rank_results_and_failures() -> None:
    manifest = build_cache_manifest(
        summaries=[
            RankCacheSummary(rank=0, assigned=3, succeeded=2, skipped=1),
            RankCacheSummary(
                rank=1,
                assigned=2,
                succeeded=1,
                failures=(CacheFailure(token="bad", log_name="log", error="broken"),),
            ),
        ],
        total_tokens=5,
        world_size=2,
        dry_run=False,
    )

    assert manifest["status"] == "failed"
    assert manifest["assigned"] == 5
    assert manifest["succeeded"] == 3
    assert manifest["skipped"] == 1
    assert manifest["failed"] == 1
    assert manifest["failures"] == [{"token": "bad", "log_name": "log", "error": "broken"}]


@pytest.mark.skipif(
    os.environ.get("RUN_DISTRIBUTED_CACHE_SMOKE") != "1",
    reason="requires permission to open a local TCPStore socket",
)
def test_two_process_gloo_coordination_smoke() -> None:
    environment = dict(os.environ)
    environment["LATENT_WAM_DIST_BACKEND"] = "gloo"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            "--module",
            "scripts.smoke_distributed_cache",
        ],
        cwd=os.getcwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "dry_run"' in result.stdout
