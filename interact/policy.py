from abc import ABC, abstractmethod
from typing import Dict, Union, List

import gym
import numpy as np


class Policy(ABC):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space):
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def _step(self,
              obs: np.ndarray,
              states: Union[np.ndarray, None] = None) -> Dict[str, Union[float, np.ndarray]]:
        pass

    def step(self,
             obs: np.ndarray,
             states: Union[np.ndarray, None] = None) -> Dict[str, Union[float, np.ndarray]]:
        data = self._step(obs, states)

        assert 'actions' in data, f'Dictionary returned by `_step` must contain the key "actions"'
        if states is not None:
            assert 'states' in data, 'If states are provided, dictionary returned by ' \
                                     f'`_step` must contain the key "states"'

        return data

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass


class RandomPolicy(Policy):

    def _step(self,
              obs: np.ndarray,
              states: Union[np.ndarray, None] = None) -> Dict[str, Union[float, np.ndarray, List]]:
        return {'actions': [self.action_space.sample() for _ in range(len(obs))]}

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
