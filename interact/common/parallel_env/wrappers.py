"""Wrappers for parallelized environments.

Author: Ryan Strauss
"""
import numpy as np

from interact.common.parallel_env.base import ParallelEnvWrapper
from interact.common.statistics import RunningMeanVariance


class ParallelEnvNormalizeWrapper(ParallelEnvWrapper):
    """This wrapper normalized the observations and returns from an environment.

    Normalization is done by maintaining a running mean and standard deviation of the observations and returns.

    Args:
        env: the parallelized environment to be wrapped
        normalize_obs: boolean indicating whether or not observations should be normalized
        normalize_returns: boolean indicating whether or not returns should be normalized
        clip_obs: the maximum absolute value for an observation
        clip_returns: the maximum absolute value for a return
        gamma: the discount factor
        epsilon: an epsilon used for numerical stability
    """

    def __init__(self, env, normalize_obs=True, normalize_returns=True, clip_obs=10., clip_returns=10., gamma=0.99,
                 epsilon=1e-8):
        super().__init__(env)
        self.obs_runner = RunningMeanVariance(shape=self.observation_space.shape) if normalize_obs else None
        self.ret_runner = RunningMeanVariance(shape=()) if normalize_returns else None
        self.clip_obs = clip_obs
        self.clip_returns = clip_returns
        self.gamma = gamma
        self.epsilon = epsilon
        self.returns = np.zeros(self.num_envs)

    def reset(self):
        self.returns = np.zeros(self.num_envs)
        obs = self._env.reset()
        return self._compute_obs(obs)

    def step_wait(self):
        obs, rewards, dones, infos = self._env.step_wait()
        self.returns *= self.gamma + rewards

        obs = self._compute_obs(obs)
        rewards = self._compute_rewards(rewards)

        self.returns[dones] = 0.

        return obs, rewards, dones, infos

    def _compute_obs(self, obs):
        if self.obs_runner:
            self.obs_runner.update(obs)
            obs = np.clip((obs - self.obs_runner.mean) / np.sqrt(self.obs_runner.var + self.epsilon),
                          -self.clip_obs,
                          self.clip_obs)

        return obs

    def _compute_rewards(self, rewards):
        if self.ret_runner:
            self.ret_runner.update(self.returns)
            rewards = np.clip(rewards / np.sqrt(self.ret_runner.var + self.epsilon),
                              -self.clip_returns,
                              self.clip_returns)

        return rewards
