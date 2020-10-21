from abc import ABC, abstractmethod
from typing import Dict, Callable

import gym
import tensorflow as tf


class Agent(ABC, tf.Module):

    def __init__(self, env_fn: Callable[[], gym.Env]):
        super().__init__()
        self._env_fn = env_fn

    def make_env(self):
        return self._env_fn()

    @abstractmethod
    @property
    def timesteps_per_iteration(self):
        pass

    @abstractmethod
    def train(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def act(self, obs, state=None):
        pass
