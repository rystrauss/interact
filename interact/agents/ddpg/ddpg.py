from typing import Tuple, Dict, List, Callable, Optional

import gin
import gym
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.utils import register
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.policies.actor_critic import DeterministicActorCriticPolicy
from interact.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from interact.typing import TensorType


@gin.configurable(name_or_fn="ddpg", denylist=["env_fn"])
@register("ddpg")
class DDPGAgent(Agent):
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        network: str = "mlp",
        critic_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        learning_starts: int = 1500,
        tau: float = 0.001,
        gamma: float = 0.95,
        buffer_size: int = 50000,
        batch_size: int = 256,
        num_workers: int = 1,
        num_envs_per_worker: int = 1,
        prioritized_replay: bool = False,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta: float = 0.4,
        final_prioritized_replay_beta: float = 4.0,
        prioritized_replay_beta_steps: Optional[int] = None,
        prioritized_replay_epsilon: float = 1e-6,
    ):
        super().__init__(env_fn)

        env = self.make_env()
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "DDPG can only be used with continuous action spaces."

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_beta_steps = prioritized_replay_beta_steps
        self.final_prioritized_replay_beta = final_prioritized_replay_beta
        self.prioritized_replay_epsilon = prioritized_replay_epsilon

        def policy_fn():
            return DeterministicActorCriticPolicy(
                env.observation_space, env.action_space, network
            )

        self.actor_critic = policy_fn()
        self.actor_critic.build([None, *env.observation_space.shape])

        self.runner = None
        self.runner_config = dict(
            env_fn=env_fn,
            policy_fn=policy_fn,
            num_envs_per_worker=num_envs_per_worker,
            num_workers=num_workers,
        )

        self.replay_buffer = None
        self.actor_optimizer = None
        self.critic_optimizer = None

    @property
    def timesteps_per_iteration(self) -> int:
        pass

    def pretrain_setup(self, total_timesteps: int):
        self.runner = Runner(**self.runner_config)

        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)

        if self.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.buffer_size, self.prioritized_replay_alpha
            )
        else:
            self.replay_buffer = ReplayBuffer(self.buffer_size)

    def act(self, obs: TensorType, deterministic: bool = True) -> TensorType:
        pass

    def _update(self, obs, actions, rewards, dones, next_obs, weights):
        q_values = self.actor_critic.q_function([obs, actions])

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        episodes, ep_infos = self.runner.run(
            1,
            # In the beginning, randomly select actions from a uniform distribution
            # for better exploration.
            uniform_sample=(
                update * self.timesteps_per_iteration <= self.learning_starts
            ),
        )

        self.replay_buffer.add(episodes.to_sample_batch())

        metrics = dict()
        if (
            update * self.timesteps_per_iteration > self.learning_starts
            and update % self.train_freq == 0
        ):
            if self.prioritized_replay:
                sample = self.replay_buffer.sample(
                    self.batch_size, self.beta_schedule(update)
                )
                weights = sample[SampleBatch.PRIO_WEIGHTS]
            else:
                sample = self.replay_buffer.sample(self.batch_size)
                weights = 1.0

            metrics.update(self._update(update, sample, weights))

        return metrics, ep_infos
