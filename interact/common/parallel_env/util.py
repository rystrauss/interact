import gym

from interact.common.parallel_env.subprocess_parallel_env import SubprocessParallelEnv
from interact.common.wrappers import Monitor


def make_parallelized_env(env_id, num_workers, seed):
    def make_env(rank):
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env)
            return env

        return _thunk

    return SubprocessParallelEnv([make_env(i) for i in range(num_workers)])
