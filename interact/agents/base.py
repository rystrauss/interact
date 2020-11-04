from abc import ABC, abstractmethod
from typing import Dict, Callable, Tuple, List

import gym
import tensorflow as tf


class Agent(ABC, tf.Module):

    def __init__(self, env_fn: Callable[[], gym.Env]):
        super().__init__()
        self._env_fn = env_fn

    @property
    @abstractmethod
    def timesteps_per_iteration(self):
        pass

    @abstractmethod
    def act(self, obs, state=None):
        pass

    def make_env(self):
        return self._env_fn()

    @abstractmethod
    def train(self) -> Tuple[Dict[str, float], List[Dict]]:
        pass

    def setup(self, total_timesteps):
        pass
