from typing import List

import numpy as np

from interact.experience.postprocessing import Postprocessor
from interact.experience.sample_batch import SampleBatch


class EpisodeBatch:
    """Container for multiple `SampleBatch`s of episodes.

    Useful for apply postprocessing to many episodes.

    Note that this class is not meant to be instantiated directly.
    """

    def __init__(self, **kwargs):
        if not kwargs.get('internal'):
            raise ValueError('This class is only meant to be directly instantiated internally.')

        self._episodes = kwargs.get('episodes')

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, item):
        return self._episodes[item]

    @classmethod
    def from_episodes(cls, episodes: List[SampleBatch]) -> "EpisodeBatch":
        """Creates a new EpisodeBatch from a list of SampleBatches.

        Args:
            episodes: A list of `SampleBatch` objects, each of which contains data from a single episode.

        Returns:
            A new EpisodeBatch.
        """
        return cls(episodes=episodes, internal=True)

    def for_each(self, postprocessor: Postprocessor):
        """Applies the given postprocessor to all episodes in the batch.

        Args:
            postprocessor: A postprocessor to apply to the episodes in the batch.

        Returns:
            None
        """
        for ep in self._episodes:
            postprocessor.apply(ep)

    def to_sample_batch(self) -> SampleBatch:
        """Converts this episodes batch to a single SampleBatch containing all of the episodes' experience.

        Returns:
            A new SampleBatch containing all of the episodes' data.
        """
        if len(self._episodes) == 1:
            return self._episodes[0]

        merged_data = {}

        for key in self._episodes[0].keys():
            merged_data[key] = np.concatenate([t[key] for t in self._episodes], axis=0)

        merged_batch = SampleBatch(merged_data, _finished=True)
        return merged_batch
