import numpy as np


class SampleBatch:
    OBS = 'obs'
    ACTIONS = 'actions'
    REWARDS = 'rewards'
    DONES = 'dones'
    INFOS = 'infos'

    ACTION_LOGP = 'action_logp'
    VALUE_PREDS = 'value_preds'

    RETURNS = 'returns'

    def __init__(self, *args, finished=False, **kwargs):
        self._data = dict(*args, **kwargs)
        self._finished = finished

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value

    @property
    def is_finished(self):
        return self._finished

    def add(self, **kwargs):
        if self._finished:
            raise RuntimeError('Cannot add to a trajectory that has already been finished.')

        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []

            self._data[key].append(value)

    def apply(self, transformation):
        transformation(self)
        return self

    def keys(self):
        return self._data.keys()

    def finish(self):
        if self._finished:
            return

        for key in self._data.keys():
            self._data[key] = np.asarray(self._data[key], dtype=np.float32).swapaxes(0, 1)

        self._finished = True

    @staticmethod
    def stack(batches):
        stacked_data = {}

        assert all([b.is_finished for b in batches]), 'All batches to be stacked must be finished.'

        for key in batches[0].keys():
            stacked_data[key] = np.concatenate([t[key] for t in batches], axis=0)

        stacked_batch = SampleBatch(stacked_data, finished=True)
        return stacked_batch
