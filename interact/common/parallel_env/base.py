from abc import ABC, abstractmethod

from gym import Env


class ParallelEnv(Env, ABC):

    def __init__(self, num_envs, observation_space, action_space):
        self._num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @property
    def num_envs(self):
        return self._num_envs

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()
