from typing import List

import numpy as np


class SampleBatch:
    """Represents a batch of environment experience.

    Loosely based on a similar class from RLLib:
    https://github.com/ray-project/ray/blob/master/rllib/policy/sample_batch.py
    """

    OBS = 'obs'
    NEXT_OBS = 'next_obs'
    ACTIONS = 'actions'
    REWARDS = 'rewards'
    DONES = 'dones'
    INFOS = 'infos'

    ACTION_LOGP = 'action_logp'
    VALUE_PREDS = 'value_preds'

    RETURNS = 'returns'
    ADVANTAGES = 'advantages'

    EPS_ID = 'eps_id'

    def __init__(self, *args, **kwargs):
        self._finished = kwargs.pop('_finished', False)
        self._data = dict(*args, **kwargs)

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __contains__(self, item):
        return item in self._data

    def __str__(self):
        return str(self._data)

    @property
    def is_finished(self):
        return self._finished

    def add(self, **kwargs):
        """Adds data to this batch."""
        if self._finished:
            raise RuntimeError('Cannot add to a trajectory that has already been finished.')

        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []

            self._data[key].append(value)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def extract_episodes(self) -> List["SampleBatch"]:
        """Parses the batch to partition experience by episodes.

        Returns:
            A list of samples batches where each batch contains only data from a single episode.
        """
        assert not self._finished, 'Cannot extract episodes from a finished sample batch.'

        for key in self._data.keys():
            self._data[key] = np.asarray(self._data[key], dtype=np.float32).swapaxes(0, 1)

        slices = []
        for j, row in enumerate(self._data[SampleBatch.EPS_ID]):
            start = 0
            end = 1
            for i in range(len(row)):
                if i == len(row) - 1 or row[start] != row[end]:
                    slices.append(SampleBatch({k: v[j, start:end] for k, v in self._data.items()}, _finished=True))
                    start = end

                end += 1

        assert len(slices) == len(np.unique(self._data[SampleBatch.EPS_ID]))

        return slices

    def shuffle(self) -> "SampleBatch":
        """Shuffles the data in the batch while being consistent across keys.

        Returns:
            self
        """
        assert self._finished, 'Trying to shuffle an unfinished sample batch.'

        sizes = [len(v) for v in self._data.values()]
        assert all(s == sizes[0] for s in sizes), \
            'All values in the sample batch must have the same length in order to shuffle.'

        inds = np.random.choice(sizes[0], (sizes[0],), replace=False)
        for key, value in self._data.items():
            self._data[key] = value[inds]

        return self
