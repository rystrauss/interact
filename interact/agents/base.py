from abc import ABC, abstractmethod
from typing import Dict, Callable, Tuple, List

import gym
import tensorflow as tf

from interact.typing import TensorType


class Agent(ABC, tf.Module):
    """Abstract class that all agents must inherit from.

    Args:
        env_fn: A function that, when called, returns an instance of the agent's environment.
    """

    def __init__(self, env_fn: Callable[[], gym.Env]):
        super().__init__()
        self._env_fn = env_fn

    @property
    @abstractmethod
    def timesteps_per_iteration(self) -> int:
        """Returns the number of environment timesteps that are executed upon each call of the `train` method."""
        pass

    @abstractmethod
    def act(self, obs: TensorType, state: List[TensorType] = None) -> TensorType:
        """Computes actions for the provided observation.

        Args:
            obs: The environment observations.
            state: An optional list of recurrent states.

        Returns:
            The agent's actions for the given observations.
        """
        pass

    def make_env(self) -> gym.Env:
        """Makes a new copy of this agent's environment."""
        return self._env_fn()

    @abstractmethod
    def train(self) -> Tuple[Dict[str, float], List[Dict]]:
        """Performs a single iteration of training for the agent.

        The definition of what exactly constitutes one iteration (e.g. how many gradient updates) can
        vary from agent to agent.

        Returns:
            metrics: A dictionary of values that should be logged during training.
            ep_infos: A list of dictionaries contain info about any episodes which may have been completed during
                the training iteration.
        """
        pass

    def setup(self, total_timesteps: int):
        """Performs any necessary setup before training begins.

        This method is always called once before the first call the `train`. Can be overridden to allow for
        agent specific setup that depends on outside information. For example, this is useful for properly
        setting learning rate schedules based on the total number of training steps.

        Args:
            total_timesteps: The total number of environment timesteps for which the agent is about to be trained.

        Returns:
            None.
        """
        pass
