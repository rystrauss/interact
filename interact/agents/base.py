from abc import ABC, abstractmethod
from typing import Dict

import tensorflow as tf


class Agent(ABC, tf.Module):

    def __init__(self, env: str):
        super().__init__()
        self._env = env

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
