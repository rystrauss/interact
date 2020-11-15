import numpy as np

from interact.experience.sample_batch import SampleBatch


class ReplayBuffer:
    """A buffer for storing and sampling experience.

    This buffer operates as a circular buffer, where the oldest experience is overwritten once the
    size limit is reached.

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
            num_items: Number of items to sample from this buffer.

        Returns:
            SampleBatch: concatenated batch of items.
        """
        idxes = np.random.randint(0, len(self._storage), (num_items,))
        return SampleBatch.concat_samples([self._storage[i] for i in idxes])
