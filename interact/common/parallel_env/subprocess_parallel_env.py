import multiprocessing as mp
from collections import OrderedDict

import gym
import numpy as np
from gym.vector.utils import CloudpickleWrapper

from interact.common.parallel_env.base import ParallelEnv
from interact.logger import printc, Colors


def _flatten_obs(obs, space):
    assert isinstance(obs, (list, tuple)), 'expected list or tuple of observations per environment'
    assert len(obs) > 0, 'need observations from at least one environment'

    if isinstance(space, gym.spaces.Dict):
        assert isinstance(space.spaces, OrderedDict), 'Dict space must have ordered subspaces'
        assert isinstance(obs[0], dict), 'non-dict observation for environment with Dict observation space'
        return OrderedDict([(k, np.stack([o[k] for o in obs])) for k in space.spaces.keys()])
    elif isinstance(space, gym.spaces.Tuple):
        assert isinstance(obs[0], tuple), 'non-tuple observation for environment with Tuple observation space'
        obs_len = len(space.spaces)
        return tuple((np.stack([o[i] for o in obs]) for i in range(obs_len)))
    else:
        return np.stack(obs)


def _worker(index, remote, parent_remote, env_fn_wrapper):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

    parent_remote.close()
    env = env_fn_wrapper.fn()

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(step_env(env, data))
            elif cmd == 'seed':
                remote.send(env.seed(data))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                printc(Colors.BLUE, f'SubprocessParallelEnv worker {index}: closed')
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        printc(Colors.YELLOW, f'SubprocessParallelEnv worker {index}: received KeyboardInterrupt')
    finally:
        remote.close()


class SubprocessParallelEnv(ParallelEnv):

    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        num_envs = len(env_fns)

        forkserver_available = 'forkserver' in mp.get_all_start_methods()
        start_method = 'forkserver' if forkserver_available else 'spawn'

        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe(duplex=True) for _ in range(num_envs)])

        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(zip(self.work_remotes, self.remotes, env_fns)):
            args = (i, work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(num_envs, observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))

        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        obs = [remote.recv() for remote in self.remotes]
        return _flatten_obs(obs, self.observation_space)

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        if self.closed:
            return

        if self.waiting:
            for remote in self.remotes:
                remote.recv()

        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()

        self.closed = True
