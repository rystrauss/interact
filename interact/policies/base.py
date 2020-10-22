from abc import ABC, abstractmethod
from typing import Dict, Union, List

import gym
import numpy as np
import tensorflow as tf

from interact.experience.sample_batch import SampleBatch


class Policy(ABC, tf.keras.layers.Layer):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
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


class RandomPolicy(Policy):

    def _step(self,
              obs: np.ndarray,
              states: Union[np.ndarray, None] = None) -> Dict[str, Union[float, np.ndarray, List]]:
        return {
            SampleBatch.ACTIONS: np.array([self.action_space.sample() for _ in range(len(obs))]),
            SampleBatch.VALUE_PREDS: np.random.random(len(obs))
        }
