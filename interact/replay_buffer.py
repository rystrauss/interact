import numpy as np

from interact.experience.sample_batch import SampleBatch


class ReplayBuffer:
    def __init__(self, size: int):
        """Create Prioritized Replay buffer.
        Args:
            size (int): Max number of timesteps to store in the FIFO buffer.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, item: SampleBatch):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            self._storage[self._next_idx] = item

        # Wrap around storage as a circular buffer once we hit maxsize.
        if self._next_idx >= self._maxsize:
            self._next_idx = 0
        else:
            self._next_idx += 1

    def sample(self, num_items: int) -> SampleBatch:
        """Sample a batch of experiences.

        Args:
            num_items (int): Number of items to sample from this buffer.

        Returns:
            SampleBatchType: concatenated batch of items.
        """
        idxes = np.random.randint(0, len(self._storage), (num_items,))
        return SampleBatch.concat_samples([self._storage[i] for i in idxes])
