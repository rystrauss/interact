import numpy as np
import pytest

from interact.environments.utils import make_env_fn
from interact.experience.sample_batch import SampleBatch
from interact.experience.runner import Runner
from interact.replay_buffer import ReplayBuffer
from interact.tests.mock_policy import MockPolicy


@pytest.fixture
def cartpole_episode_batch():
    env_fn = make_env_fn("CartPole-v1")
    env = env_fn()
    policy_fn = lambda: MockPolicy(91, env.observation_space, env.action_space)

    runner = Runner(env_fn, policy_fn, seed=91)
    episodes, _ = runner.run(8)
    return episodes, policy_fn


def test_replay_buffer_sample(cartpole_episode_batch):
    episodes, policy_fn = cartpole_episode_batch

    batch = episodes.to_sample_batch()

    buffer = ReplayBuffer(4)

    buffer.add(batch)

    assert len(buffer) == 4

    sample = buffer.sample(4)

    for i in range(4):
        found_match = False
        for j, b in enumerate(batch[SampleBatch.OBS][-4:]):
            if np.array_equal(sample[SampleBatch.OBS][i], b):
                found_match = True
                break
        assert found_match

        for k in sample.keys():
            assert np.array_equal(sample[k][i], batch[k][-4:][j])
