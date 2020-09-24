"""Utilities for the command line interface.

Author: Ryan Strauss
"""

import os
import re
from collections import defaultdict

import gym

from interact.common.atari_wrappers import wrap_atari
from interact.common.parallel_env import SubprocessParallelEnv
from interact.common.parallel_env.dummy_parallel_env import DummyParallelEnv
from interact.common.parallel_env.wrappers import ParallelEnvNormalizeWrapper
from interact.common.wrappers import MonitorEpisodeWrapper, ClipActionsWrapper


def extract_extra_kwargs(context_args):
    """Extracts keyword arguments from the command line args.

    Args:
        context_args: The arguments captured by the command line context.

    Returns:
        A dictionary of keyword arguments.
    """

    def pairwise(iterable):
        iterator = iter(iterable)
        return zip(iterator, iterator)

    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    kwargs = {k.lstrip('--'): parse(v) for k, v in pairwise(context_args)}
    return kwargs


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


def make_env(env_id, env_type, seed=None, rank=0, video_callable=None, video_path=None):
    """Makes a Gym environment and automatically performs some setup depending on the environment type.

    For example, this function will automatically wrap Atari envs with standard modifications.

    Args:
        env_id: The ID of the environment to make.
        env_type: The type of the environment.
        seed: The base random seed to seed the environment with.
        rank: The rank of the environment, to ensure that separate workers have different seeds.
        video_callable: A function that accepts an episode ID and returns True if that episode should be saved to video.
            If None, no episodes will be saved.
        video_path: The path to where episode videos will be saved. Must be provided if `video_callable` is provided.

    Returns:
        The environment.
    """
    env = gym.make(env_id)
    if seed is not None:
        env.seed(seed + rank)

    if video_callable is not None:
        assert video_path is not None, 'video_path must be provided whenever video_callable is provided'
        env = gym.wrappers.Monitor(env, os.path.join(video_path, f'env_{rank}'), video_callable=video_callable)

    env = MonitorEpisodeWrapper(env)

    if env_type == 'atari':
        env = wrap_atari(env)

    if isinstance(env.action_space, gym.spaces.Box):
        env = ClipActionsWrapper(env)

    return env


def make_parallelized_env(env_id, num_workers, seed=None, normalize_obs=False, normalize_rewards=False,
                          video_callable=None, video_path=None):
    """Creates a parallelized version of a gym environment.

    Args:
        env_id: The id of the gym environment being created.
        num_workers: The number of workers to execute the environment in parallel.
        seed: The random seed used to initialize the parallelized environment.
        normalize_obs: If true, normalization will be applied to observations.
        normalize_rewards: If true, normalization will be applied to rewards.
        video_callable: A function that accepts an episode ID and returns a boolean indicating whether or not a video
            of that episode should be saved.
        video_path: The directory in which videos should be saved. Must be provided if `video_callable` is provided.

    Returns:
        A new `ParallelEnv` that executes the provided gym environment.
    """
    env_type = get_env_type(env_id)

    def get_make_env_fn(rank):
        return lambda: make_env(env_id, env_type, seed=seed, rank=rank, video_callable=video_callable,
                                video_path=video_path)

    parallel_env_type = DummyParallelEnv if num_workers == 1 else SubprocessParallelEnv
    env = parallel_env_type([get_make_env_fn(i) for i in range(num_workers)])

    if normalize_obs or normalize_rewards:
        env = ParallelEnvNormalizeWrapper(env, normalize_obs, normalize_rewards)

    return env
