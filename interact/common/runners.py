from abc import ABC, abstractmethod

import numpy as np
from gym import Env

from interact.common.parallel_env.base import ParallelEnv
from interact.common.policies import Policy


class AbstractRunner(ABC):

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
                dones: whether or not the episode is finished
                actions: the actions
                values: the value function output
                neglogpacs: the negative log probabilities
        """
        pass

    @property
    def steps(self):
        return self._steps

    @property
    def batch_size(self):
        return self.nsteps * self.num_env
