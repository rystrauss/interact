from abc import ABC, abstractmethod
from typing import Dict, Union, List

import gym
import numpy as np
import tensorflow as tf

from interact.experience.sample_batch import SampleBatch


class Policy(ABC, tf.keras.layers.Layer):
    """An abstract class representing a policy.

    A policy is able to receive environment observations and produce actions
    (as well as other information used for training).
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, obs: np.ndarray, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        """Computes policy information for the given observation.

        Args:
            obs: A state observation for which policy information should be computed.

        Returns:
            A dictionary that is guaranteed to contain 'actions', and can optionally
            contain other useful policy information.
        """
        data = self._step(obs, **kwargs)

        assert (
            SampleBatch.ACTIONS in data
        ), f'Dictionary returned by `_step` must contain the key "actions"'

        return data

    @abstractmethod
    def _step(self, obs: np.ndarray, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        """Implements the specific behavior of `step` for child policy classes."""
        pass


class RandomPolicy(Policy):
    """A toy policy which takes random actions."""

    def _step(
        self, obs: np.ndarray, **kwargs
    ) -> Dict[str, Union[float, np.ndarray, List]]:
        return {
            SampleBatch.ACTIONS: np.array(
                [self.action_space.sample() for _ in range(len(obs))]
            ),
        }
