from enum import Enum

import numpy as np


class Experience(Enum):
    OBS = 'obs'
    ACTIONS = 'actions'
    REWARDS = 'rewards'
    DONES = 'dones'
    INFOS = 'infos'

    STATES = 'states'

    LOGPACS = 'logpacs'
    VALUES = 'values'

    def __str__(self):
        return str(self.value)


class Trajectory:

    def __init__(self):
        self._data = dict()
        self._finished = False

    def __getitem__(self, item):
        return self._data[item]

    def add(self, **kwargs):
        if self._finished:
            raise RuntimeError('Cannot add to a trajectory that has already been finished.')

        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = []

            self._data[key].append(value)

    def finish_trajectory(self):
        if self._finished:
            raise RuntimeError('Trying to finish a trajectory that has already been finished.')

        for key in self._data.keys():
            self._data[key] = np.asarray(self._data[key], dtype=np.float32)

        self._finished = True
