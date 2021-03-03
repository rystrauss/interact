"""Replay buffers for storing experience.

This module is adapted from RLLib:
https://github.com/ray-project/ray/blob/master/rllib/execution/replay_buffer.py
"""

from typing import Optional, List

import numpy as np

from interact.experience.sample_batch import SampleBatch
from interact.utils.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    """A buffer for storing and sampling experience.

    This buffer operates as a circular buffer, where the oldest experience is
    overwritten once the size limit is reached.

    Args:
        size: The maximum size of the buffer.
        seed: Random seed used by this buffer's PRNG.
    """

    def __init__(self, size: int, seed: Optional[int] = None):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, batch: SampleBatch):
        """Adds experience to the replay buffer.

        Args:
            batch: A sample batch of experience to be added.

        Returns:
            None.
        """
        for item in batch.split():
            self._add_item(item)

    def _add_item(self, item: SampleBatch):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            self._storage[self._next_idx] = item

        # Wrap around storage as a circular buffer once we hit maxsize.
        if self._next_idx >= self._maxsize - 1:
            self._next_idx = 0
        else:
            self._next_idx += 1

    def sample(self, num_items: int) -> SampleBatch:
        """Sample a batch of experiences.

        Args:
            num_items: Number of items to sample from this buffer.

        Returns:
            Concatenated batch of items.
        """
        idxes = self._rng.randint(0, len(self._storage), (num_items,))
        return SampleBatch.concat_samples([self._storage[i] for i in idxes])


class PrioritizedReplayBuffer(ReplayBuffer):
    """A prioritized replay buffer.

    Args:
        size: The maximum size of the buffer.
        alpha: The amount of prioritization that is used
            (0 - no prioritization, 1 - full prioritization).
        seed: Random seed used by this buffer's PRNG.
    """

    def __init__(self, size: int, alpha: float, seed: Optional[int] = None):
        super().__init__(size, seed)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, batch: SampleBatch):
        """Adds experience to the replay buffer.

        If the sample batch contains values for `SampleBatch.PRIO_WEIGHTS`, they will
        be used as the priority weights of the samples.

        Args:
            batch: A sample batch of experience to be added.

        Returns:
            None.
        """
        for item in batch.split():
            idx = self._next_idx
            self._add_item(item)

            weight = item.get(SampleBatch.PRIO_WEIGHTS)
            if weight is None:
                weight = self._max_priority

            self._it_sum[idx] = weight ** self._alpha
            self._it_min[idx] = weight ** self._alpha

    def _sample_proportional(self, num_items: int) -> List[int]:
        res = []
        for _ in range(num_items):
            mass = self._rng.random() * self._it_sum.sum(0, len(self._storage))
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, num_items: int, beta: float) -> SampleBatch:
        """Sample a batch of experiences and return priority weights and indices.

        Args:
            num_items: Number of items to sample from this buffer.
            beta: To what degree to use importance weights
                (0 - no corrections, 1 - full correction).

        Returns:
            Concatenated batch of items including "weights"
            and "batch_indexes" fields denoting IS of each sampled
            transition and original idxes in buffer of sampled experiences.
        """
        assert beta >= 0.0

        idxes = self._sample_proportional(num_items)

        weights = []
        batch_indexes = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-beta)
            weights.append(weight / max_weight)
            batch_indexes.append(idx)

        batch = SampleBatch.concat_samples([self._storage[i] for i in idxes])

        batch[SampleBatch.PRIO_WEIGHTS] = np.array(weights)
        batch["batch_indices"] = np.array(batch_indexes)

        return batch

    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities of sampled transitions.

        Sets priority of transition at index idxes[i] in buffer to priorities[i].

        Args:
            indices: Indices of the transitions to update.
            priorities: Priorities corresponding to transitions at the sampled indices
                denoted by `indices`.
        """
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
