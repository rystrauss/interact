from typing import List, Generator

import numpy as np


class SampleBatch:
    """Represents a batch of environment experience.

    Loosely based on a similar class from RLLib:
    https://github.com/ray-project/ray/blob/master/rllib/policy/sample_batch.py
    """

    OBS = "obs"
    NEXT_OBS = "next_obs"
    ACTIONS = "actions"
    REWARDS = "rewards"
    DONES = "dones"
    INFOS = "infos"

    ACTION_LOGP = "action_logp"
    POLICY_LOGITS = "policy_logits"
    VALUE_PREDS = "value_preds"

    RETURNS = "returns"
    ADVANTAGES = "advantages"

    PRIO_WEIGHTS = "weights"

    EPS_ID = "eps_id"

    def __init__(self, *args, **kwargs):
        self._finished = kwargs.pop("_finished", False)
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
            raise RuntimeError(
                "Cannot add to a trajectory that has already been finished."
            )

        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []

            self._data[key].append(value)

    def get(self, item, *args):
        return self._data.get(item, *args)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def extract_episodes(self) -> List["SampleBatch"]:
        """Parses the batch to partition experience by episodes.

        Returns:
            A list of samples batches where each batch contains only data from a single episode.
        """
        assert (
            not self._finished
        ), "Cannot extract episodes from a finished sample batch."

        for key in self._data.keys():
            self._data[key] = np.asarray(self._data[key], dtype=np.float32).swapaxes(
                0, 1
            )

        slices = []
        for j, row in enumerate(self._data[SampleBatch.EPS_ID]):
            start = 0
            end = 1
            for i in range(len(row)):
                if i == len(row) - 1 or row[start] != row[end]:
                    slices.append(
                        SampleBatch(
                            {k: v[j, start:end] for k, v in self._data.items()},
                            _finished=True,
                        )
                    )
                    start = end

                end += 1

        assert len(slices) == len(np.unique(self._data[SampleBatch.EPS_ID]))

        return slices

    def to_minibatches(
        self, num_minibatches: int
    ) -> Generator["SampleBatch", None, None]:
        assert (
            self._finished
        ), "Trying to produces minibatches from an unfinished sample batch."

        sizes = [len(v) for v in self._data.values()]
        assert all(
            s == sizes[0] for s in sizes
        ), "All values in the sample batch must have the same length in order to produce minibatches."

        batch_size = sizes[0]
        minibatch_size = batch_size // num_minibatches

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            minibatch = SampleBatch(
                {k: v[start:end] for k, v in self._data.items()}, _finished=True
            )
            yield minibatch

    def shuffle(self) -> "SampleBatch":
        """Shuffles the data in the batch while being consistent across keys.

        Returns:
            self
        """
        assert self._finished, "Trying to shuffle an unfinished sample batch."

        sizes = [len(v) for v in self._data.values()]
        assert all(
            s == sizes[0] for s in sizes
        ), "All values in the sample batch must have the same length in order to shuffle."

        inds = np.random.permutation(sizes[0])
        for key, value in self._data.items():
            self._data[key] = value[inds]

        return self

    def split(self) -> Generator["SampleBatch", None, None]:
        sizes = [len(v) for v in self._data.values()]
        return self.to_minibatches(sizes[0])

    @staticmethod
    def concat_samples(samples: List["SampleBatch"]) -> "SampleBatch":
        merged = {}

        for batch in samples:
            for k, v in batch.items():
                if k not in merged:
                    merged[k] = []

                merged[k].append(v)

        for k, v in merged.items():
            merged[k] = np.concatenate(v, axis=0)

        return SampleBatch(merged, _finished=True)
