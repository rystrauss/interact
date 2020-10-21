from abc import ABC, abstractmethod
from typing import Dict, Union

import gym
import numpy as np

from interact.experience import Experience


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

        assert Experience.ACTIONS in data, f'Dictionary returned by `_step` must contain the key "{Experience.ACTIONS}"'
        if states is not None:
            assert Experience.STATES in data, 'If states are provided, dictionary returned by ' \
                                              f'`_step` must contain the key "{Experience.STATES}"'

        return data

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass
