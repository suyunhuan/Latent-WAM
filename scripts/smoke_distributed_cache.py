"""Run a two-or-more-process coordination smoke without loading model weights."""

import json

from navsim.agents.latent_wam.distributed_cache import DistributedCacheContext, plan_shard_ids


def main() -> None:
    """Verify process-group setup, shard ownership, and object gathering."""
    context = DistributedCacheContext.from_environment(cuda_available=False)
    with context.activate():
        owned_shards = plan_shard_ids(5, context.world_size, context.rank)
        ranks = context.all_gather_object({"rank": context.rank, "shards": owned_shards})
        if context.rank == 0:
            print(json.dumps({"world_size": context.world_size, "ranks": ranks}, indent=2))
        context.barrier()


if __name__ == "__main__":
    main()
