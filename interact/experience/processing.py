from typing import Callable

import numpy as np

from interact.experience.sample_batch import SampleBatch


def discount_with_dones(rewards, dones, gamma):
    """Given a sequence of rewards, calculates the corresponding sequence of returns.

    Args:
        rewards: A sequence of rewards, where each element is the reward received at that time step.
        dones: A sequence of booleans that indicate whether or not an episode finished at that time step.
        gamma: The discount factor.

    Returns:
        A sequence of the corresponding returns.
    """
    discounted = []
    ret = 0

    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)
        discounted.append(ret)

    return np.array(discounted[::-1])


def compute_returns(gamma=0.99, last_values=None) -> Callable[[SampleBatch], None]:
    def _batch_fn(batch):
        returns_ = np.zeros_like(batch[SampleBatch.REWARDS])
        rewards_ = batch[SampleBatch.REWARDS]
        dones_ = batch[SampleBatch.DONES]
        last_values_ = last_values or np.zeros_like(batch[SampleBatch.REWARDS][:, 0])

        for i, (rewards, dones, value) in enumerate(zip(rewards_, dones_, last_values_)):
            rewards = rewards.tolist()
            dones = dones.tolist()

            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, gamma)

            returns_[i] = rewards

        batch[SampleBatch.RETURNS] = returns_

    return _batch_fn
