import copy
import itertools
import os
import uuid
from typing import Callable, List, Tuple

import numpy as np
import ray
from gym import Env
from gym.spaces import Box
from gym.vector import SyncVectorEnv

from interact.experience.episode_batch import EpisodeBatch
from interact.experience.sample_batch import SampleBatch
from interact.policies.base import Policy
from interact.typing import TensorType


class Worker:
    """A worker that is responsible for executing a policy in an environment and collecting experience.

    Args:
        env_fn: A function that returns a Gym environment when called. The returned environment is used to collect
            experience.
        policy_fn: A function that returns a Policy when called. The returned Policy is used to collect
            experience.
        num_envs: The number of environments to synchronously execute within this worker.
    """

    def __init__(self, env_fn: Callable[[], Env], policy_fn: Callable[[], Policy], num_envs: int = 1):
        self.env = SyncVectorEnv([env_fn] * num_envs)
        self.env.seed(int.from_bytes(os.urandom(4), byteorder='big'))

        self.policy = policy_fn()

        self.obs = np.zeros(self.env.observation_space.shape,
                            dtype=self.env.observation_space.dtype)
        self.obs[:] = self.env.reset()
        self.dones = [False for _ in range(self.env.num_envs)]
        self.eps_ids = [uuid.uuid4().int for _ in range(self.env.num_envs)]

    def collect(self, num_steps: int = 1) -> Tuple[List[SampleBatch], List[dict]]:
        """Executes the policy in the environment and returns the collected experience.

        Args:
            num_steps: The number of environment steps to execute in each synchronous environment.

        Returns:
            episodes: A list of `SamplesBatch`s, where each batch contains experience from a single episode
                (each of which may or may not be a complete episode).
            ep_infos: A list of dictionaries containing information about any episodes which were completed
                during collection.
        """
        batch = SampleBatch()
        ep_infos = []

        for _ in range(num_steps):
            data = self.policy.step(self.obs)

            data[SampleBatch.OBS] = self.obs.copy()
            data[SampleBatch.EPS_ID] = copy.copy(self.eps_ids)

            batch.add(**data)

            clipped_actions = data[SampleBatch.ACTIONS].numpy()
            if isinstance(self.env.action_space, Box):
                clipped_actions = np.clip(clipped_actions, self.env.action_space.low, self.env.action_space.high)

            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            for i, done in enumerate(self.dones):
                if done:
                    self.eps_ids[i] = uuid.uuid4().int

            batch.add(**{
                SampleBatch.REWARDS: rewards,
                SampleBatch.DONES: self.dones,
                SampleBatch.NEXT_OBS: self.obs.copy()
            })

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

        return batch.extract_episodes(), ep_infos

    def update_policy(self, weights: List[TensorType]):
        """Updates the weights of this worker's policy.

        Args:
            weights: A list of weights to be applied to the policy.

        Returns:
            None
        """
        self.policy.set_weights(weights)


@ray.remote
class RemoteWorker(Worker):
    """A remote version of `Worker`."""
    pass


class Runner:
    """Responsible for collecting experience from an environment.

    This class is uses an arbitrary number of `Worker`s to execute a policy in an environment and
    aggregate the collected experience.

    Args:
        env_fn: A function that returns a Gym environment when called. The returned environment is used to collect
            experience.
        policy_fn: A function that returns a Policy when called. The returned Policy is used to collect
            experience.
        num_envs_per_worker: The number of environments to synchronously execute within each worker.
        num_workers: THe number of parallel workers to use for experience collection.
    """

    def __init__(self,
                 env_fn: Callable[[], Env],
                 policy_fn: Callable[[], Policy],
                 num_envs_per_worker: int = 1,
                 num_workers: int = 1):
        if num_workers == 1:
            self._workers = [Worker(env_fn, policy_fn, num_envs_per_worker)]
        else:
            self._workers = [RemoteWorker.remote(env_fn, policy_fn, num_envs_per_worker) for _ in range(num_workers)]

    def run(self, num_steps: int = 1) -> Tuple[EpisodeBatch, List[dict]]:
        """Executes the policy in the environment and returns the collected experience.

        Args:
            num_steps: The number of steps to take in each environment.

        Returns:
            episodes: An `EpisodeBatch` containing the collected data.
            ep_infos: A list of dictionaries containing information about any episodes which were completed
                during collection.
        """
        if len(self._workers) == 1:
            episodes, ep_infos = self._workers[0].collect(num_steps)
            return EpisodeBatch.from_episodes(episodes), ep_infos

        episodes, ep_infos = zip(*ray.get([w.collect.remote(num_steps) for w in self._workers]))
        ep_infos = list(itertools.chain.from_iterable(ep_infos))
        episodes = list(itertools.chain.from_iterable(episodes))

        return EpisodeBatch.from_episodes(episodes), ep_infos

    def update_policies(self, weights: List[TensorType]):
        """Updates the weights of this runner's policy.

        Args:
            weights: A list of weights to be applied to the policy.

        Returns:
            None
        """
        if len(self._workers) == 1:
            self._workers[0].update_policy(weights)
        else:
            ray.get([w.update_policy.remote(weights) for w in self._workers])
