import numpy as np

from interact.experience.sample_batch import SampleBatch
from interact.policies.base import Policy


class MockPolicy(Policy):

    def _step(self,
              obs,
              states=None,
              **kwargs):
        return {
            SampleBatch.ACTIONS: np.array([np.random.choice([0, 1])] * len(obs)),
            SampleBatch.VALUE_PREDS: np.array([100.0] * len(obs))
        }

    def value(self, inputs, **kwargs):
        return np.array([100.0] * len(inputs))
