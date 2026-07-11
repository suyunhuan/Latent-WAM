from navsim.agents.latent_wam.sampler import LocalityAwareDistributedSampler


def test_locality_sampler_keeps_shard_buckets_contiguous() -> None:
    sampler = LocalityAwareDistributedSampler(
        locality_ids=[0, 1, 0, 2, 1, 2],
        shuffle=True,
        seed=7,
        rank=0,
        world_size=1,
    )

    order = list(sampler)
    bucket_sequence = [[0, 1, 0, 2, 1, 2][index] for index in order]

    assert sorted(order) == list(range(6))
    assert (
        len([bucket_sequence[index] for index in range(1, 6) if bucket_sequence[index] != bucket_sequence[index - 1]])
        == 2
    )


def test_locality_sampler_balances_disjoint_ddp_indices() -> None:
    samplers = [
        LocalityAwareDistributedSampler(
            locality_ids=[0, 0, 1, 1, 2, 2],
            shuffle=True,
            seed=3,
            rank=rank,
            world_size=2,
        )
        for rank in range(2)
    ]

    rank_indices = [list(sampler) for sampler in samplers]

    assert len(rank_indices[0]) == len(rank_indices[1]) == 3
    assert set(rank_indices[0]).isdisjoint(rank_indices[1])
    assert sorted(rank_indices[0] + rank_indices[1]) == list(range(6))


def test_global_mode_is_deterministic_per_epoch_and_changes_between_epochs() -> None:
    sampler = LocalityAwareDistributedSampler(
        locality_ids=[0, 0, 1, 1, 2, 2],
        mode="global",
        shuffle=True,
        seed=11,
    )
    epoch_zero = list(sampler)
    sampler.set_epoch(1)
    epoch_one = list(sampler)

    assert epoch_zero != epoch_one
    sampler.set_epoch(0)
    assert list(sampler) == epoch_zero
