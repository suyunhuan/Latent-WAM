"""Run a two-or-more-process coordination smoke without loading model weights."""

import json

from navsim.agents.latent_wam.distributed_cache import (
    DistributedCacheContext,
    RankCacheSummary,
    build_cache_manifest,
    canonical_tokens,
    plan_token_shard,
)


def main() -> None:
    """Verify process-group setup, token agreement, sharding, and summary gathering."""
    context = DistributedCacheContext.from_environment(cuda_available=False)
    tokens = canonical_tokens(["echo", "alpha", "delta", "bravo", "charlie"])
    with context.activate():
        context.verify_token_set(tokens)
        shard = plan_token_shard(tokens, context.world_size, context.rank)
        summary = RankCacheSummary(rank=context.rank, assigned=len(shard))
        summaries = context.all_gather_object(summary)
        manifest = build_cache_manifest(
            summaries=summaries,
            total_tokens=len(tokens),
            world_size=context.world_size,
            dry_run=True,
        )
        if context.rank == 0:
            print(json.dumps(manifest, indent=2))
        context.barrier()


if __name__ == "__main__":
    main()
