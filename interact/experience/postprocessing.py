from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from scipy import signal

from interact.experience.sample_batch import SampleBatch
from interact.typing import TensorType


def discount_cumsum(x: np.ndarray, gamma: float) -> float:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    This implementation comes from RLLib:
    https://github.com/ray-project/ray/blob/61e41257e7b5aa15d10e3e968516906d94bdf30a/rllib/evaluation/postprocessing.py#L7

    Args:
        gamma: The discount factor gamma.

    Returns:
        float: The discounted cumulative sum over the reward sequence `x`.
    """
    return signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def compute_advantages(
    rollout: SampleBatch,
    last_r: float,
    gamma: float = 0.9,
    lam: float = 1.0,
    use_gae: bool = True,
    use_critic: bool = True,
) -> SampleBatch:
    """Given a rollout, compute its value targets and the advantages.

    This implementation comes from RLLib:
    https://github.com/ray-project/ray/blob/61e41257e7b5aa15d10e3e968516906d94bdf30a/rllib/evaluation/postprocessing.py#L30

    Args:
        rollout: SampleBatch of a single trajectory.
        last_r: Value estimation for last observation.
        gamma: Discount factor.
        lambda: Parameter for GAE.
        use_gae: Using Generalized Advantage Estimation.
        use_critic: Whether to use critic (value estimates). Setting this to False will
            use 0 as baseline.

    Returns:
        The modified SampleBatch which has been updated with advantages.
    """
    rollout_size = len(rollout[SampleBatch.ACTIONS])

    assert (
        SampleBatch.VALUE_PREDS in rollout or not use_critic
    ), "use_critic=True but values not found"
    assert use_critic or not use_gae, "Can't use GAE without using a value function"

    if use_gae:
        vpred_t = np.concatenate([rollout[SampleBatch.VALUE_PREDS], np.array([last_r])])
        delta_t = rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[SampleBatch.ADVANTAGES] = discount_cumsum(delta_t, gamma * lam)
        rollout[SampleBatch.RETURNS] = (
            rollout[SampleBatch.ADVANTAGES] + rollout[SampleBatch.VALUE_PREDS]
        ).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS], np.array([last_r])]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if use_critic:
            rollout[SampleBatch.ADVANTAGES] = (
                discounted_returns - rollout[SampleBatch.VALUE_PREDS]
            )
            rollout[SampleBatch.RETURNS] = discounted_returns
        else:
            rollout[SampleBatch.ADVANTAGES] = discounted_returns
            rollout[SampleBatch.RETURNS] = np.zeros_like(
                rollout[SampleBatch.ADVANTAGES]
            )

    rollout[SampleBatch.ADVANTAGES] = rollout[SampleBatch.ADVANTAGES].astype(np.float32)

    assert all(
        val.shape[0] == rollout_size for key, val in rollout.items()
    ), "Rollout stacked incorrectly!"
    return rollout


class Postprocessor(ABC):
    """An abstract class representing a postprocessing transformation."""

    @abstractmethod
    def apply(self, episode: SampleBatch):
        """Applied this postprocessing transformation to the given `SampleBatch`.

        The batch is assumed to contain a single episode.

        Args:
            episode: The `SampleBatch` to be processed.

        Returns:
            None
        """
        pass


class AdvantagePostprocessor(Postprocessor):
    """A postprocessor which computes advantages for an episode.

    Generalized Advantage Estimation can optionally be used to compute advantages.

    Args:
        value_fn: A function that take observations as input and returns the
            corresponding value estimates. This is used for bootstrapping.
        gamma: The discount factor.
        lam: The lambda parameter used in GAE.
        use_gae: Whether or not to use GAE.
        use_critic: Whether to use critic (value estimates). Setting this to False will
            use 0 as baseline.
    """

    def __init__(
        self,
        value_fn: Callable[[TensorType], TensorType],
        gamma: float = 0.99,
        lam: float = 0.95,
        use_gae: bool = True,
        use_critic: bool = True,
    ):
        self.value_fn = value_fn
        self.gamma = gamma
        self.lam = lam
        self.use_gae = use_gae
        self.use_critic = use_critic

    def apply(self, episode: SampleBatch):
        if episode[SampleBatch.DONES][-1]:
            last_r = 0.0
        else:
            last_r = np.asarray(self.value_fn(episode[SampleBatch.NEXT_OBS][-1:])[0])

        compute_advantages(
            episode, last_r, self.gamma, self.lam, self.use_gae, self.use_critic
        )
