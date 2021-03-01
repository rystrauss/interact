from typing import Optional

import numpy as np

from interact.experience.sample_batch import SampleBatch
from interact.utils.segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    """A buffer for storing and sampling experience.

    This buffer operates as a circular buffer, where the oldest experience is
    overwritten once the size limit is reached.

    Args:
        size: The maximum size of the buffer.
    """

    def __init__(self, size: int):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, item: SampleBatch):
        """Adds experience to the replay buffer.

        Args:
            item: A sample batch of experience to be added.

        Returns:
            None.
        """
        for e in item.split():
            if self._next_idx >= len(self._storage):
                self._storage.append(e)
            else:
                self._storage[self._next_idx] = e

            # Wrap around storage as a circular buffer once we hit maxsize.
            if self._next_idx >= self._maxsize - 1:
                self._next_idx = 0
            else:
                self._next_idx += 1

    def sample(self, num_items: int, seed: Optional[int] = None) -> SampleBatch:
        """Sample a batch of experiences.

        Args:
            num_items: Number of items to sample from this buffer.
            seed: Random seed.

        Returns:
            SampleBatch: concatenated batch of items.
        """
        idxes = np.random.RandomState(seed).randint(0, len(self._storage), (num_items,))
        return SampleBatch.concat_samples([self._storage[i] for i in idxes])


class PrioritizedReplayBuffer(ReplayBuffer):
    """A prioritized replay buffer.

    Args:
        size (int): Max number of items to store in the FIFO buffer.
        alpha (float): how much prioritization is used
            (0 - no prioritization, 1 - full prioritization).
    """

    def __init__(self, size: int, alpha: float, beta: float):
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        assert beta >= 0.0
        self._alpha = alpha
        self._beta = beta

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, item: SampleBatch):
        """Adds experience to the replay buffer.

        Args:
            item: A sample batch of experience to be added. If this batch contains an
                entry for the key `SampleBatch.PRIO_WEIGHTS`, it will be used as the
                corresponding priority weights.

        Returns:
            None.
        """
        for e in item.split():
            if self._next_idx >= len(self._storage):
                self._storage.append(e)
            else:
                self._storage[self._next_idx] = e

            weight = e.get(SampleBatch.PRIO_WEIGHTS)
            if weight is None:
                weight = self._max_priority

            self._it_sum[self._next_idx] = weight ** self._alpha
            self._it_min[self._next_idx] = weight ** self._alpha

            # Wrap around storage as a circular buffer once we hit maxsize.
            if self._next_idx >= self._maxsize:
                self._next_idx = 0
            else:
                self._next_idx += 1

    def _sample_proportional(self, num_items: int, seed: Optional[int] = None):
        res = []
        for _ in range(num_items):
            mass = np.random.RandomState(seed).random() * self._it_sum.sum(
                0, len(self._storage)
            )
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, num_items: int, seed: Optional[int] = None) -> SampleBatch:
        """Sample a batch of experiences and return priority weights, indices.

        Args:
            num_items (int): Number of items to sample from this buffer.
            seed: Random seed.

        Returns:
            SampleBatchType: Concatenated batch of items including "weights"
                and "batch_indexes" fields denoting IS of each sampled
                transition and original idxes in buffer of sampled experiences.
        """
        idxes = self._sample_proportional(num_items, seed)

        weights = []
        batch_indexes = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-self._beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self._storage)) ** (-self._beta)
            weights.append(weight / max_weight)
            batch_indexes.append(idx)

        batch = SampleBatch.concat_samples([self._storage[i] for i in idxes])

        batch[SampleBatch.PRIO_WEIGHTS] = np.array(weights)
        batch["batch_indices"] = np.array(batch_indexes)

        return batch

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        Sets priority of transition at index idxes[i] in buffer to priorities[i].

        Args:
            idxes: [int]
              List of idxes of sampled transitions
            priorities: [float]
              List of updated priorities corresponding to
              transitions at the sampled idxes denoted by
              variable `idxes`.
        """
        assert len(idxes) == len(priorities)

        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self._storage)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)
