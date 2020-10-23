"""This module contains environment wrappers.

Author: Ryan Strauss
"""

import gym
import numpy as np


class MonitorEpisodeWrapper(gym.Wrapper):
    """A wrapper that monitors episode data from an environment and makes it available in the info dict.

    When an episode has finished, an 'episode' entry is put into the info dict that contains the episode's
    length ('length') and total reward ('reward').

    Args:
        env: The environment being wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

        self.rewards = None

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)

        if done:
            ep_rew = sum(self.rewards)
            eplen = len(self.rewards)
            ep_info = {'reward': round(ep_rew, 6), 'length': eplen}
            info['episode'] = ep_info

        return observation, reward, done, info


class ClipActionsWrapper(gym.Wrapper):
    """Clips actions to the allowed range.

    This should only be used with environments that have a `Box` action space.
    """

    def step(self, action):
        action = np.nan_to_num(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
