from abc import ABC, abstractmethod

import numpy as np
from scipy import signal

from interact.experience.sample_batch import SampleBatch
from interact.policies.actor_critic import ActorCriticPolicy


def discount_cumsum(x: np.ndarray, gamma: float) -> float:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma (float): The discount factor gamma.

    Returns:
        float: The discounted cumulative sum over the reward sequence `x`.
    """
    return signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def compute_advantages(rollout: SampleBatch,
                       last_r: float,
                       gamma: float = 0.9,
                       lam: float = 1.0,
                       use_gae: bool = True,
                       use_critic: bool = True):
    rollout_size = len(rollout[SampleBatch.ACTIONS])

    assert SampleBatch.VALUE_PREDS in rollout or not use_critic, 'use_critic=True but values not found'
    assert use_critic or not use_gae, 'Can\'t use gae without using a value function'

    if use_gae:
        vpred_t = np.concatenate([rollout[SampleBatch.VALUE_PREDS], np.array([last_r])])
        delta_t = rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[SampleBatch.ADVANTAGES] = discount_cumsum(delta_t, gamma * lam)
        rollout[SampleBatch.RETURNS] = \
            (rollout[SampleBatch.ADVANTAGES] + rollout[SampleBatch.VALUE_PREDS]).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate([rollout[SampleBatch.REWARDS], np.array([last_r])])
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(np.float32)

        if use_critic:
            rollout[SampleBatch.ADVANTAGES] = discounted_returns - rollout[SampleBatch.VALUE_PREDS]
            rollout[SampleBatch.RETURNS] = discounted_returns
        else:
            rollout[SampleBatch.ADVANTAGES] = discounted_returns
            rollout[SampleBatch.RETURNS] = np.zeros_like(rollout[SampleBatch.ADVANTAGES])

    rollout[SampleBatch.ADVANTAGES] = rollout[SampleBatch.ADVANTAGES].astype(np.float32)

    assert all(val.shape[0] == rollout_size for key, val in rollout.items()), 'Rollout stacked incorrectly!'
    return rollout


class Postprocessor(ABC):

    @abstractmethod
    def apply(self, episode: SampleBatch):
        pass


class AdvantagePostprocessor(Postprocessor):

    def __init__(self, policy, gamma=0.99, lam=0.95, use_gae=True, use_critic=True):
        assert isinstance(policy, ActorCriticPolicy)
        self.policy = policy
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.use_critic = use_critic

    def apply(self, episode: SampleBatch):
        if episode[SampleBatch.DONES][-1]:
            last_r = 0.0
        else:
            last_r = self.policy.value(episode[SampleBatch.NEXT_OBS][-1:]).numpy()[0]

        compute_advantages(
            episode,
            last_r,
            self.gamma,
            self.lam,
            self.use_gae,
            self.use_critic)


class EpisodeBatch:

    def __init__(self, **kwargs):
        if kwargs.get('internal') != True:
            raise ValueError('This class is only meant to be directly instantiated internally.')

        self._episodes = kwargs.get('episodes')

    @classmethod
    def from_episodes(cls, episodes):
        return cls(episodes=episodes, internal=True)

    def for_each(self, postprocessor: Postprocessor):
        for ep in self._episodes:
            postprocessor.apply(ep)

    def to_sample_batch(self):
        stacked_data = {}

        for key in self._episodes[0].keys():
            if key not in {SampleBatch.NEXT_OBS, SampleBatch.NEXT_DONES}:
                stacked_data[key] = np.concatenate([t[key] for t in self._episodes], axis=0)

        stacked_batch = SampleBatch(stacked_data, _finished=True)
        return stacked_batch
