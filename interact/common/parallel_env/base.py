"""Contains the base for parallelized environments.

Author: Ryan Strauss
"""

from abc import ABC, abstractmethod

from gym import Env


class ParallelEnv(Env, ABC):
    """A parallelized version of a Gym environment.

    Some number of environments are executed in parallel and each step of this environment returns the experience from
    all of the workers.

    Args:
        num_envs: The number of environments to be executed in parallel.
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self._num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    def num_envs(self):
        """The number of parallel environments."""
        return self._num_envs

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        """Asynchronously steps each of the parallel environments.

        Args:
            actions: A list of actions to be submitted to each of the workers.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """Waits for each of the worker environments to finish stepping then returns the results.

        Returns:
            A 4-tuple with the following:
                observations: a list of the observations from all of the workers
                rewards: a list of the rewards from all of the workers
                dones: a list of the done indicators from all of the workers
                infos: a list of the info dicts from all of the workers
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
