import numpy as np
import pytest

from interact.experience.postprocessing import AdvantagePostprocessor
from interact.experience.sample_batch import SampleBatch


@pytest.fixture
def cartpole_episode_batch():
    from interact.environments.utils import make_env_fn
    from interact.experience.runner import Runner
    from interact.tests.mock_policy import MockPolicy

    np.random.seed(91)
    env_fn = make_env_fn('CartPole-v1')
    env = env_fn()
    policy_fn = lambda: MockPolicy(env.observation_space, env.action_space)

    runner = Runner(env_fn, policy_fn, seed=91)
    episodes, _ = runner.run(15)
    return episodes, policy_fn


def test_advantage_postprocessor(cartpole_episode_batch):
    episodes, policy_fn = cartpole_episode_batch

    episodes.for_each(AdvantagePostprocessor(policy_fn(), gamma=0.9, use_critic=False, use_gae=False))

    assert episodes[0][SampleBatch.ADVANTAGES][-1] == 1
    assert episodes[1][SampleBatch.ADVANTAGES][-1] == 91

    episodes.for_each(AdvantagePostprocessor(policy_fn(), gamma=1.0, use_critic=False, use_gae=False))

    assert episodes[0][SampleBatch.ADVANTAGES][-1] == 1
    assert episodes[1][SampleBatch.ADVANTAGES][-1] == 101
