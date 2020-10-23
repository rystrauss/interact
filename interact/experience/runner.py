import itertools
import os

import numpy as np
import ray
from gym.spaces import Box
from gym.vector import SyncVectorEnv

from interact.experience.sample_batch import SampleBatch


class Worker:

    def __init__(self, env_fn, policy_fn, num_envs=1):
        self.env = SyncVectorEnv([env_fn] * num_envs)
        self.env.seed(int.from_bytes(os.urandom(4), byteorder='big'))

        self.policy = policy_fn()

        self.obs = np.zeros(self.env.observation_space.shape,
                            dtype=self.env.observation_space.dtype)
        self.obs[:] = self.env.reset()
        self.dones = [False for _ in range(self.env.num_envs)]

    def collect(self, num_steps=1):
        batch = SampleBatch()
        ep_infos = []

        for _ in range(num_steps):
            data = self.policy.step(self.obs)

            data[SampleBatch.OBS] = self.obs.copy()
            data[SampleBatch.DONES] = self.dones

            batch.add(**data)

            clipped_actions = data[SampleBatch.ACTIONS].numpy()
            if isinstance(self.env.action_space, Box):
                clipped_actions = np.clip(clipped_actions, self.env.action_space.low, self.env.action_space.high)

            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            batch.add(rewards=rewards)

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

        batch.add(last_obs=self.obs.copy(), last_dones=self.dones)

        batch.finish()

        return batch, ep_infos

    def update_policy(self, weights):
        self.policy.set_weights(weights)


@ray.remote
class RemoteWorker(Worker):
    pass


class Runner:

    def __init__(self, env_fn, policy_fn, num_envs_per_worker=1, num_workers=1):
        if num_workers == 1:
            self._workers = [Worker(env_fn, policy_fn, num_envs_per_worker)]
        else:
            self._workers = [RemoteWorker.remote(env_fn, policy_fn, num_envs_per_worker) for _ in range(num_workers)]

    def run(self, num_steps=1):
        if len(self._workers) == 1:
            return self._workers[0].collect(num_steps)

        batches, ep_infos = zip(*ray.get([w.collect.remote(num_steps) for w in self._workers]))
        ep_infos = list(itertools.chain.from_iterable(ep_infos))

        return SampleBatch.stack(batches), ep_infos

    def update_policies(self, weights):
        if len(self._workers) == 1:
            self._workers[0].update_policy(weights)
        else:
            ray.get([w.update_policy.remote(weights) for w in self._workers])
