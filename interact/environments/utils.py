import re
from collections import defaultdict
from typing import Callable

import gin
import gym

from interact.environments.atari_wrappers import wrap_atari
from interact.environments.wrappers import (
    ClipActionsWrapper,
    MonitorEpisodeWrapper,
    ScaleRewardsWrapper,
)


def get_env_type(env_id: str) -> str:
    """Determines an environment's type.

    Examples of types would be "atari" and "mujoco".

    Args:
        env_id: The environment ID whose type is to be determines.

    Returns:
        The environment type of `env_id`.
    """
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(":")[0].split(".")[-1]
        _game_envs[env_type].add(env.id)

    if env_id in _game_envs.keys():
        env_type = env_id
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ":" in env_id:
            env_type = re.sub(r":.*", "", env_id)
        assert env_type is not None, "env_id {} is not recognized in env types".format(
            env_id, _game_envs.keys()
        )

    return env_type


@gin.configurable("env", blacklist=["env_id", "seed"])
def make_env_fn(
    env_id: str,
    seed: int = None,
    reward_scale: float = None,
    episode_time_limit: int = None,
) -> Callable[[], gym.Env]:
    """Returns a function that constructs the given environment when called.

    Also ensures that relevant wrappers will be applied to the environment.

    Args:
        env_id: The ID of the environment to be created by the returned function.
        seed: An optional seed to seed the returned environment with.
        reward_scale: Factor by which rewards should be scaled.

    Returns:
        A function that returns a new instance of the requested environment when called.
    """
    env_type = get_env_type(env_id)

    def _env_fn():
        env = gym.make(env_id)

        if episode_time_limit is not None:
            env._max_episode_steps = episode_time_limit

        if seed is not None:
            env.seed(seed)

        env = MonitorEpisodeWrapper(env)

        if reward_scale is not None:
            env = ScaleRewardsWrapper(env, reward_scale)

        if env_type == "atari":
            env = wrap_atari(env)

        if isinstance(env.action_space, gym.spaces.Box):
            env = ClipActionsWrapper(env)

        return env

    return _env_fn
