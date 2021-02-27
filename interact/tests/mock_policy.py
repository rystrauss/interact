import gym
import numpy as np

from interact.experience.sample_batch import SampleBatch
from interact.policies.base import Policy


class MockPolicy(Policy):
    def __init__(
        self, seed: int, observation_space: gym.Space, action_space: gym.Space
    ):
        super().__init__(observation_space, action_space)
        self._rng = np.random.RandomState(seed)

    def _step(self, obs, states=None, **kwargs):
        return {
            SampleBatch.ACTIONS: np.array([self._rng.choice([0, 1])] * len(obs)),
            SampleBatch.VALUE_PREDS: np.array([100.0] * len(obs)),
        }

    def value(self, inputs, **kwargs):
        return np.array([100.0] * len(inputs))
