import numpy as np


class SampleBatch:
    OBS = 'obs'
    NEXT_OBS = 'next_obs'
    ACTIONS = 'actions'
    REWARDS = 'rewards'
    DONES = 'dones'
    NEXT_DONES = 'next_dones'
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
        if self._finished:
            raise RuntimeError('Cannot add to a trajectory that has already been finished.')

        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []

            self._data[key].append(value)

    def apply(self, transformation):
        assert self._finished
        transformation(self)
        return self

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def extract_episodes(self):
        if self._finished:
            return

        for key in self._data.keys():
            self._data[key] = np.asarray(self._data[key], dtype=np.float32).swapaxes(0, 1)

        slices = []
        for j, row in enumerate(self._data[SampleBatch.EPS_ID]):
            start = 0
            end = 1
            for i in range(len(row)):
                if i == len(row) - 1 or row[start] != row[end]:
                    slices.append(SampleBatch({k: v[j, start:end] for k, v in self._data.items()}, _finished=True))

                    if i == len(row) - 1:
                        slices[-1]._data[SampleBatch.NEXT_OBS] = self._data[SampleBatch.NEXT_OBS][j]
                        slices[-1]._data[SampleBatch.NEXT_DONES] = self._data[SampleBatch.NEXT_DONES][j]

                    start = end

                end += 1

        assert len(slices) == len(np.unique(self._data[SampleBatch.EPS_ID]))

        return slices

    def shuffle(self):
        assert self._finished

        sizes = [len(v) for v in self._data.values()]
        assert all(s == sizes[0] for s in sizes), \
            'All values in the sample batch must have the same length in order to shuffle.'

        inds = np.random.choice(sizes[0], (sizes[0],), replace=False)
        for key, value in self._data.items():
            self._data[key] = value[inds]

        return self
