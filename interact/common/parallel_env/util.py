"""Utilities for working with parallelized environments.

Author: Ryan Strauss
"""

import gym

from interact.common.parallel_env.subprocess_parallel_env import SubprocessParallelEnv
from interact.common.wrappers import Monitor


def make_parallelized_env(env_id, num_workers, seed):
    """Creates a parallelized version of a gym environment.

    Args:
        env_id: The id of the gym environment being created.
        num_workers: The number of workers to execute the environment in parallel.
        seed: The random seed used to initialize the parallelized environment.

    Returns:
        A new `ParallelEnv` that executes the provided gym environment.
    """

    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env)
            return env

        return _thunk

    return SubprocessParallelEnv([make_env(i) for i in range(num_workers)])
