import gym
import numpy as np
import ray
from gym.spaces import Box
from gym.vector import SyncVectorEnv

from interact.experience.sample_batch import SampleBatch
from interact.policy import RandomPolicy


class Worker:

    def __init__(self, env_fn, policy, num_envs=1):
        self.env = SyncVectorEnv([env_fn] * num_envs)
        self.policy = policy

        self.obs = np.zeros(self.env.observation_space.shape,
                            dtype=self.env.observation_space.dtype)
        self.obs[:] = self.env.reset()
        self.dones = [False for _ in range(self.env.num_envs)]

    def collect(self, num_steps=1):
        trajectory = SampleBatch()
        ep_infos = []

        for _ in range(num_steps):
            data = self.policy.step(self.obs)

            data[SampleBatch.OBS] = self.obs.copy()
            data[SampleBatch.DONES] = self.dones

            trajectory.add(**data)

            clipped_actions = data[SampleBatch.ACTIONS]
            if isinstance(self.env.action_space, Box):
                clipped_actions = np.clip(clipped_actions, self.env.action_space.low, self.env.action_space.high)

            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            trajectory.add(rewards=rewards)

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

        trajectory.finish()

        return trajectory, ep_infos


@ray.remote
class RemoteWorker(Worker):
    pass


class Runner:

    def __init__(self, env_fn, policy, num_envs_per_worker=1, num_workers=1):
        if num_workers == 1:
            self._workers = [Worker(env_fn, policy, num_envs_per_worker)]
        else:
            self._workers = [RemoteWorker.remote(env_fn, policy, num_envs_per_worker) for _ in range(num_workers)]

    def run(self, num_steps=1):
        if len(self._workers) == 1:
            return self._workers[0].collect(num_steps)

        batches, ep_infos = zip(*ray.get([w.collect.remote(num_steps) for w in self._workers]))
        return SampleBatch.stack(batches)


if __name__ == '__main__':
    ray.init()


    def make_env():
        return gym.make('CartPole-v1')


    env = make_env()

    policy = RandomPolicy(env.observation_space, env.action_space)

    runner = Runner(make_env, policy, num_envs_per_worker=2, num_workers=3)

    batch = runner.run(32)

    for key in batch.keys():
        print(key, batch[key].shape)
