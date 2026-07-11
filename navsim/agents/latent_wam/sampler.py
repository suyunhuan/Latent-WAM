"""Locality-aware deterministic sampling for sharded Latent WAM caches."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterator, Sequence

from torch.utils.data import Sampler


class LocalityAwareDistributedSampler(Sampler[int]):
    """Shuffle shard-local buckets and distribute balanced indices.

    Args:
        locality_ids: Primary shard ID for every dataset sample.
        mode: ``locality`` block shuffle or ``global`` random shuffle.
        shuffle: Whether to randomize order.
        seed: Base deterministic shuffle seed.
        rank: Global distributed rank.
        world_size: Number of distributed ranks.
        drop_last: Whether to discard a non-divisible tail instead of padding.
    """

    def __init__(
        self,
        locality_ids: Sequence[int],
        mode: str = "locality",
        shuffle: bool = True,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        drop_last: bool = False,
    ) -> None:
        if mode not in ("locality", "global"):
            raise ValueError("sampler mode must be 'locality' or 'global'")
        if world_size < 1 or rank < 0 or rank >= world_size:
            raise ValueError("invalid distributed sampler rank")
        self.locality_ids = list(locality_ids)
        self.mode = mode
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.drop_last = drop_last
        self.epoch = 0
        if drop_last:
            self.num_samples = len(self.locality_ids) // world_size
        else:
            self.num_samples = math.ceil(len(self.locality_ids) / world_size) if self.locality_ids else 0

    def set_epoch(self, epoch: int) -> None:
        """Set the deterministic shuffle epoch.

        Args:
            epoch: Zero-based training epoch.
        """
        self.epoch = epoch

    def _global_order(self) -> list[int]:
        generator = random.Random(self.seed + self.epoch)
        if self.mode == "global":
            order = list(range(len(self.locality_ids)))
            if self.shuffle:
                generator.shuffle(order)
            return order

        buckets: dict[int, list[int]] = defaultdict(list)
        for index, locality_id in enumerate(self.locality_ids):
            buckets[locality_id].append(index)
        bucket_ids = sorted(buckets)
        if self.shuffle:
            generator.shuffle(bucket_ids)
            for bucket in buckets.values():
                generator.shuffle(bucket)
        return [index for bucket_id in bucket_ids for index in buckets[bucket_id]]

    def __iter__(self) -> Iterator[int]:
        order = self._global_order()
        total_size = self.num_samples * self.world_size
        if self.drop_last:
            order = order[:total_size]
        elif order and len(order) < total_size:
            order.extend(order[: total_size - len(order)])
        return iter(order[self.rank : total_size : self.world_size])

    def __len__(self) -> int:
        return self.num_samples
