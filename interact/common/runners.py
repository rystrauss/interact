"""This module provides the framework for collecting experience from parallel environments.

Author: Ryan Strauss
"""

from abc import ABC, abstractmethod

import numpy as np
from gym import Env

from interact.common.parallel_env.base import ParallelEnv
from interact.common.policies import Policy


class AbstractRunner(ABC):
    """The base class from which all runners inherit.

    A runner is an object which is provided with an environment and a policy and has a single `run` method which,
    when called, returns a batch of experience collected in the environment by following the given policy.

    Args:
        env: The Gym environment from which experience will be collected.
        policy: The policy that will be used to collect experience.
        nsteps: The number of steps to be taken in the environment on each call to `run`.
    """

    def __init__(self, env, policy, nsteps):
        assert isinstance(env, Env)
        assert isinstance(policy, Policy)

        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.num_env = env.num_envs if isinstance(env, ParallelEnv) else 1
        self.batch_ob_shape = (self.num_env * self.nsteps, *env.observation_space.shape)
        self.obs = np.zeros((self.num_env, *env.observation_space.shape), dtype=env.observation_space.dtype.name)
        self.obs[:] = env.reset()
        self.dones = [False for _ in range(self.num_env)]

        self._steps = 0

    @abstractmethod
    def run(self):
        """Collects experience from the environment.

        Returns:
            A 5-tuple with the following:
                obs: the environment observations
                returns: the returns
                actions: the actions
                values: the value function output
                infos: the info dicts returned by the environments
        """
        pass

    @property
    def steps(self):
        """The total number of environment steps that have been executed."""
        return self._steps

    @property
    def batch_size(self):
        """The size of a batch returned by `run`."""
        return self.nsteps * self.num_env
