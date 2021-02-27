import copy
import itertools
import os
import uuid
from typing import Callable, List, Tuple

import numpy as np
import ray
from gym import Env
from gym.spaces import Box

from interact.environments.vector_env import VectorEnv
from interact.experience.episode_batch import EpisodeBatch
from interact.experience.sample_batch import SampleBatch
from interact.policies.base import Policy
from interact.typing import TensorType


class Worker:
    """Executes a policy in an environment and collects experience.

    Args:
        env_fn: A function that returns a Gym environment when called. The returned
            environment is used to collect experience.
        policy_fn: A function that returns a Policy when called. The returned Policy is
            used to collect experience.
        num_envs: The number of environments to synchronously execute within this
            worker.
        seed: Optional seed with which to see this worker's environments.
    """

    def __init__(
        self,
        env_fn: Callable[[], Env],
        policy_fn: Callable[[], Policy],
        num_envs: int = 1,
        seed: int = None,
    ):
        self.env = VectorEnv([env_fn] * num_envs)

        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="big")
        self.env.seed(seed)
        self.env.reset()

        self.policy = policy_fn()

        self.eps_ids = [uuid.uuid4().int for _ in range(self.env.num_envs)]

    @classmethod
    def as_remote(cls, **kwargs):
        return ray.remote(**kwargs)(cls)

    def collect(
        self, num_steps: int = 1, **kwargs
    ) -> Tuple[List[SampleBatch], List[dict]]:
        """Executes the policy in the environment and returns the collected experience.

        Args:
            num_steps: The number of environment steps to execute in each synchronous
                environment.

        Returns:
            episodes: A list of `SamplesBatch`s, where each batch contains experience
                from a single episode (each of which may or may not be a complete
                episode).
            ep_infos: A list of dictionaries containing information about any episodes
                which were completed during collection.
        """
        batch = SampleBatch()
        ep_infos = []

        for _ in range(num_steps):
            data = self.policy.step(self.env.observations, **kwargs)

            data[SampleBatch.OBS] = self.env.observations.copy()
            data[SampleBatch.EPS_ID] = copy.copy(self.eps_ids)

            clipped_actions = np.asarray(data[SampleBatch.ACTIONS])
            if isinstance(self.env.action_space, Box):
                clipped_actions = np.clip(
                    clipped_actions,
                    self.env.action_space.low,
                    self.env.action_space.high,
                )

            next_obs, rewards, dones, infos = self.env.step(clipped_actions)

            for i, done in enumerate(dones):
                if done:
                    self.eps_ids[i] = uuid.uuid4().int

            # This ensures that terminations which occurred due to the episode time
            # limit are not interpreted as environment terminations. This effectively
            # implements the partial bootstrapping method described in
            # https://arxiv.org/abs/1712.00378
            for i, info in enumerate(infos):
                if info.get("TimeLimit.truncated", False):
                    dones[i] = False
                    next_obs[i] = info.pop("TimeLimit.next_obs")

            data[SampleBatch.REWARDS] = rewards
            data[SampleBatch.DONES] = dones
            data[SampleBatch.NEXT_OBS] = next_obs
            batch.add(data)

            for info in infos:
                maybe_ep_info = info.get("episode")
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


class Runner:
    """Responsible for collecting experience from an environment.

    This class is uses an arbitrary number of `Worker`s to execute a policy in an
    environment and aggregate the collected experience.

    Args:
        env_fn: A function that returns a Gym environment when called. The returned
            environment is used to collect experience.
        policy_fn: A function that returns a Policy when called. The returned Policy is
            used to collect experience.
        num_envs_per_worker: The number of environments to synchronously execute within
            each worker.
        num_workers: The number of parallel workers to use for experience collection.
        seed: Optional seed with which to see this runner's environments.
    """

    def __init__(
        self,
        env_fn: Callable[[], Env],
        policy_fn: Callable[[], Policy],
        num_envs_per_worker: int = 1,
        num_workers: int = 1,
        seed: int = None,
    ):
        if num_workers == 1:
            self._workers = [Worker(env_fn, policy_fn, num_envs_per_worker, seed)]
        else:
            self._workers = [
                Worker.as_remote(num_gpus=0).remote(
                    env_fn, policy_fn, num_envs_per_worker, seed
                )
                for _ in range(num_workers)
            ]

    def run(self, num_steps: int = 1, **kwargs) -> Tuple[EpisodeBatch, List[dict]]:
        """Executes the policy in the environment and returns the collected experience.

        Args:
            num_steps: The number of steps to take in each environment.

        Returns:
            episodes: An `EpisodeBatch` containing the collected data.
            ep_infos: A list of dictionaries containing information about any episodes
                which were completed during collection.
        """
        if len(self._workers) == 1:
            episodes, ep_infos = self._workers[0].collect(num_steps, **kwargs)
            return EpisodeBatch.from_episodes(episodes), ep_infos

        episodes, ep_infos = zip(
            *ray.get([w.collect.remote(num_steps, **kwargs) for w in self._workers])
        )
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
