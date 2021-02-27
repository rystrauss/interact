import numpy as np
import pytest

from interact.environments.utils import make_env_fn
from interact.experience.sample_batch import SampleBatch
from interact.experience.runner import Runner
from interact.tests.mock_policy import MockPolicy


@pytest.fixture
def cartpole_runner():
    env_fn = make_env_fn("CartPole-v1")
    env = env_fn()
    policy_fn = lambda: MockPolicy(91, env.observation_space, env.action_space)

    return Runner(env_fn, policy_fn, seed=91)


@pytest.fixture
def time_limit_cartpole_runner():
    env_fn = make_env_fn("CartPole-v1", episode_time_limit=8)
    env = env_fn()
    policy_fn = lambda: MockPolicy(91, env.observation_space, env.action_space)

    return Runner(env_fn, policy_fn, seed=91)


def test_run(cartpole_runner):
    episodes, ep_infos = cartpole_runner.run(15)

    assert len(episodes) == 2
    assert len(ep_infos) == 1
    assert len(episodes[0][SampleBatch.OBS]) == 13
    assert len(episodes[0][SampleBatch.REWARDS]) == 13
    assert len(episodes[1][SampleBatch.OBS]) == 2
    assert len(episodes[1][SampleBatch.REWARDS]) == 2

    EXPECTED_ACTIONS = np.array([0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0])
    EXPECTED_OBS = np.array(
        [
            [-0.03233464, 0.01765816, -0.02843043, 0.0147195],
            [-0.03198147, -0.17704477, -0.02813604, 0.29829845],
            [-0.03552237, -0.37175459, -0.02217007, 0.58197668],
            [-0.04295746, -0.56655901, -0.01053054, 0.86759404],
            [-0.05428864, -0.7615361, 0.00682134, 1.15694754],
            [-0.06951936, -0.56650373, 0.02996029, 0.86641125],
            [-0.08084944, -0.76202033, 0.04728852, 1.16836153],
            [-0.09608985, -0.5675444, 0.07065575, 0.89087138],
            [-0.10744073, -0.76355014, 0.08847318, 1.20490194],
            [-0.12271174, -0.95969712, 0.11257121, 1.52394884],
            [-0.14190568, -0.76610023, 0.14305019, 1.26841763],
            [-0.15722768, -0.96272967, 0.16841854, 1.60226263],
            [-0.17648228, -1.15939741, 0.2004638, 1.94237158],
        ]
    )
    EXPECTED_NEXT_OBS = np.array(
        [
            [-0.03198147, -0.17704477, -0.02813604, 0.29829845],
            [-0.03552237, -0.37175459, -0.02217007, 0.58197668],
            [-0.04295746, -0.56655901, -0.01053054, 0.86759404],
            [-0.05428864, -0.7615361, 0.00682134, 1.15694754],
            [-0.06951936, -0.56650373, 0.02996029, 0.86641125],
            [-0.08084944, -0.76202033, 0.04728852, 1.16836153],
            [-0.09608985, -0.5675444, 0.07065575, 0.89087138],
            [-0.10744073, -0.76355014, 0.08847318, 1.20490194],
            [-0.12271174, -0.95969712, 0.11257121, 1.52394884],
            [-0.14190568, -0.76610023, 0.14305019, 1.26841763],
            [-0.15722768, -0.96272967, 0.16841854, 1.60226263],
            [-0.17648228, -1.15939741, 0.2004638, 1.94237158],
            [-0.02172156, 0.03067718, 0.02869466, -0.01193504],
        ]
    )

    np.testing.assert_equal(episodes[0][SampleBatch.ACTIONS], EXPECTED_ACTIONS)
    np.testing.assert_allclose(episodes[0][SampleBatch.OBS], EXPECTED_OBS, 1e-06)
    np.testing.assert_allclose(
        episodes[0][SampleBatch.NEXT_OBS], EXPECTED_NEXT_OBS, 1e-06
    )

    assert episodes[0][SampleBatch.DONES][-1] == 1
    assert episodes[1][SampleBatch.DONES][-1] == 0


def test_time_limit(cartpole_runner, time_limit_cartpole_runner):
    episodes, ep_infos = cartpole_runner.run(15)
    time_limit_episodes, ep_infos = time_limit_cartpole_runner.run(15)

    np.testing.assert_equal(
        episodes[0][SampleBatch.OBS][:8], time_limit_episodes[0][SampleBatch.OBS]
    )

    np.testing.assert_equal(
        episodes[0][SampleBatch.OBS][8],
        time_limit_episodes[0][SampleBatch.NEXT_OBS][-1],
    )

    np.testing.assert_equal(
        episodes[1][SampleBatch.OBS][0], time_limit_episodes[1][SampleBatch.OBS][0]
    )

    assert not time_limit_episodes[0][SampleBatch.DONES][-1]
