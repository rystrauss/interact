from copy import deepcopy

import numpy as np
import pytest

from interact.environments import make_env_fn
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.tests.mock_policy import MockPolicy


@pytest.fixture
def unfinished_batch():
    batch = SampleBatch()

    eps_id = 0
    for i in range(15):
        data = {
            SampleBatch.ACTIONS: [np.random.choice([0, 1]), np.random.choice([0, 1])],
            SampleBatch.OBS: [np.random.random((4,)), np.random.random((4,))],
            SampleBatch.EPS_ID: [eps_id, eps_id + 1],
        }

        batch.add(data)

        done = i == 12

        if done:
            eps_id += 2

        batch.add({SampleBatch.DONES: [done, done]})

    return batch


@pytest.fixture
def cartpole_episode_batch():
    env_fn = make_env_fn("CartPole-v1")
    env = env_fn()
    policy_fn = lambda: MockPolicy(91, env.observation_space, env.action_space)

    runner = Runner(env_fn, policy_fn, seed=91)
    episodes, _ = runner.run(15)
    return episodes, policy_fn


def test_extract_episodes(unfinished_batch):
    episodes = unfinished_batch.extract_episodes()

    assert len(episodes) == 4

    assert len(episodes[0][SampleBatch.OBS]) == 13
    assert len(episodes[1][SampleBatch.OBS]) == 2
    assert len(episodes[2][SampleBatch.OBS]) == 13
    assert len(episodes[3][SampleBatch.OBS]) == 2


def test_shuffle(cartpole_episode_batch):
    batch = cartpole_episode_batch[0].to_sample_batch()

    inds = np.random.RandomState(91).permutation(15)
    shuffled_batch = deepcopy(batch)
    shuffled_batch.shuffle(91)

    for k in batch.keys():
        np.testing.assert_equal(batch[k][inds], shuffled_batch[k])
