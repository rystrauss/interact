import re
from collections import defaultdict

import gym

from interact.environments.atari_wrappers import wrap_atari
from interact.environments.wrappers import ClipActionsWrapper, MonitorEpisodeWrapper


def get_env_type(env_id):
    """Determines an environment's type.

    Examples of types would be "atari" and "mujoco".

    Args:
        env_id: The environment ID whose type is to be determines.

    Returns:
        The environment type of `env_id`.
    """
    _game_envs = defaultdict(set)
    for env in gym.envs.registry.all():
        env_type = env.entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)

    if env_id in _game_envs.keys():
        env_type = env_id
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        if ':' in env_id:
            env_type = re.sub(r':.*', '', env_id)
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type


def make_env_fn(env_id, seed=None):
    env_type = get_env_type(env_id)

    def _env_fn():
        env = gym.make(env_id)
        if seed is not None:
            env.seed(seed)

        env = MonitorEpisodeWrapper(env)

        if env_type == 'atari':
            env = wrap_atari(env)

        if isinstance(env.action_space, gym.spaces.Box):
            env = ClipActionsWrapper(env)

        return env

    return _env_fn
