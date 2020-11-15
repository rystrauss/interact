import numpy as np
import pytest

from interact.environments.utils import make_env_fn
from interact.experience.sample_batch import SampleBatch
from interact.experience.runner import Runner
from interact.tests.mock_policy import MockPolicy


@pytest.fixture
def cartpole_episode_batch():
    np.random.seed(91)
    env_fn = make_env_fn('CartPole-v1')
    env = env_fn()
    policy_fn = lambda: MockPolicy(env.observation_space, env.action_space)

    runner = Runner(env_fn, policy_fn, seed=91)
    episodes, _ = runner.run(15)
    return episodes, policy_fn


def test_to_sample_batch(cartpole_episode_batch):
    episodes, policy_fn = cartpole_episode_batch

    batch = episodes.to_sample_batch()

    assert batch[SampleBatch.OBS].shape == (15, 4)
    assert batch[SampleBatch.REWARDS].shape == (15,)
    assert batch[SampleBatch.VALUE_PREDS].shape == (15,)
    assert batch[SampleBatch.DONES].shape == (15,)

    assert batch[SampleBatch.DONES][-3] == 1
