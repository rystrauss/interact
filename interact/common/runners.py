from abc import ABC, abstractmethod

from interact.common.parallel_env.base import ParallelEnv


class AbstractRunner(ABC):

    def __init__(self, env, policy, nsteps):
        self.env = env
        self.policy = policy
        self.nsteps = nsteps
        self.num_env = env.num_envs if isinstance(env, ParallelEnv) else 1

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
