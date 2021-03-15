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
from interact.schedules import LinearDecay
from interact.typing import TensorType
from interact.utils.math import polyak_update


@gin.configurable(name_or_fn="ddpg", denylist=["env_fn"])
@register("ddpg")
class DDPGAgent(Agent):
    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        network: str = "mlp",
        critic_lr: float = 1e-3,
        actor_lr: float = 1e-3,
        learning_starts: int = 1500,
        random_steps: int = 1500,
        target_update_interval: int = 1,
        tau: float = 0.002,
        gamma: float = 0.95,
        buffer_size: int = 50000,
        train_freq: int = 1,
        batch_size: int = 256,
        num_workers: int = 1,
        num_envs_per_worker: int = 1,
        prioritized_replay: bool = False,
        prioritized_replay_alpha: float = 0.6,
        prioritized_replay_beta: float = 0.4,
        final_prioritized_replay_beta: float = 4.0,
        prioritized_replay_beta_steps: Optional[int] = None,
        prioritized_replay_epsilon: float = 1e-6,
        initial_noise_scale: float = 0.1,
        final_noise_scale: float = 0.1,
        noise_scale_steps: Optional[int] = None,
        use_huber: bool = False,
        use_twin_critic: bool = False,
        policy_delay: int = 1,
        smooth_target_policy: bool = False,
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
    ):
        super().__init__(env_fn)

        env = self.make_env()
        assert isinstance(
            env.action_space, gym.spaces.Box
        ), "DDPG can only be used with continuous action spaces."

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.learning_starts = learning_starts
        self.random_steps = random_steps
        self.target_update_interval = target_update_interval
        self.tau = tau
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker
        self.prioritized_replay = prioritized_replay
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.prioritized_replay_beta = prioritized_replay_beta
        self.prioritized_replay_beta_steps = prioritized_replay_beta_steps
        self.final_prioritized_replay_beta = final_prioritized_replay_beta
        self.prioritized_replay_epsilon = prioritized_replay_epsilon
        self.initial_noise_scale = initial_noise_scale
        self.final_noise_scale = final_noise_scale
        self.noise_scale_steps = noise_scale_steps
        self.use_huber = use_huber
        self.use_twin_critic = use_twin_critic
        self.policy_delay = policy_delay
        self.smooth_target_policy = smooth_target_policy
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip

        self.actor_critic = DeterministicActorCriticPolicy(
            env.observation_space, env.action_space, network, use_twin_critic
        )
        self.target_actor_critic = DeterministicActorCriticPolicy(
            env.observation_space, env.action_space, network, use_twin_critic
        )
        self.target_actor_critic.set_weights(self.actor_critic.get_weights())

        def policy_fn():
            if num_workers == 1:
                return self.actor_critic

            return DeterministicActorCriticPolicy(
                env.observation_space, env.action_space, network, use_twin_critic
            )

        self.runner = None
        self.runner_config = dict(
            env_fn=env_fn,
            policy_fn=policy_fn,
            num_envs_per_worker=num_envs_per_worker,
            num_workers=num_workers,
        )

        self.replay_buffer = None
        self.beta_schedule = None
        self.noise_schedule = None
        self.actor_optimizer = None
        self.critic_optimizer = None

        self._has_updated = False

    @property
    def timesteps_per_iteration(self) -> int:
        return self.num_workers * self.num_envs_per_worker

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

        self.beta_schedule = LinearDecay(
            initial_learning_rate=self.prioritized_replay_beta,
            decay_steps=self.prioritized_replay_beta_steps or total_timesteps,
            end_learning_rate=self.final_prioritized_replay_beta,
        )

        self.noise_schedule = LinearDecay(
            initial_learning_rate=self.initial_noise_scale,
            decay_steps=self.noise_scale_steps or total_timesteps,
            end_learning_rate=self.initial_noise_scale,
        )

    @tf.function
    def act(self, obs: TensorType, deterministic: bool = True) -> TensorType:
        assert deterministic, "Non-deterministic actions not supported for DDPG."
        return self.actor_critic(obs)

    @tf.function
    def _update(self, obs, actions, rewards, dones, next_obs, weights, update_policy):
        target_actions = self.target_actor_critic(next_obs)
        if self.smooth_target_policy:
            epsilon = tf.random.normal(target_actions.shape, stddev=self.target_noise)
            epsilon = tf.clip_by_value(
                epsilon, -self.target_noise_clip, self.target_noise_clip
            )
            target_actions += epsilon
            target_actions = tf.clip_by_value(
                target_actions,
                self.actor_critic.action_space_low,
                self.actor_critic.action_space_high,
            )

        if self.use_twin_critic:
            target_pi_q_values = tf.minimum(
                *self.target_actor_critic.q_function([next_obs, target_actions])
            )
        else:
            target_pi_q_values = self.target_actor_critic.q_function(
                [next_obs, target_actions]
            )
        backup = rewards + self.gamma * (1 - dones) * target_pi_q_values

        loss_fn = tf.losses.huber if self.use_huber else tf.losses.mse

        with tf.GradientTape() as tape:
            if self.use_twin_critic:
                q1_values, q2_values = self.actor_critic.q_function([obs, actions])
                q1_loss = loss_fn(backup[:, tf.newaxis], q1_values[:, tf.newaxis])
                q2_loss = loss_fn(backup[:, tf.newaxis], q2_values[:, tf.newaxis])
                critic_loss = q1_loss + q2_loss
            else:
                q_values = self.actor_critic.q_function([obs, actions])
                critic_loss = loss_fn(backup[:, tf.newaxis], q_values[:, tf.newaxis])

            if not self.use_huber:
                critic_loss *= 0.5

            critic_loss = tf.reduce_mean(critic_loss * weights)

        grads = tape.gradient(
            critic_loss, self.actor_critic.q_function.trainable_variables
        )
        self.critic_optimizer.apply_gradients(
            zip(grads, self.actor_critic.q_function.trainable_variables)
        )

        if self.use_twin_critic:
            td_error = 0.5 * ((q1_values - backup) + (q2_values - backup))
        else:
            td_error = q_values - backup

        if not update_policy:
            return {"critic_loss": critic_loss}, td_error

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.actor_critic.policy.trainable_weights)

            if self.use_twin_critic:
                pi_q_values = tf.minimum(
                    *self.actor_critic.q_function([obs, self.actor_critic(obs)])
                )
            else:
                pi_q_values = self.actor_critic.q_function(
                    [obs, self.actor_critic(obs)]
                )
            actor_loss = -tf.reduce_mean(pi_q_values)

        grads = tape.gradient(actor_loss, self.actor_critic.policy.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads, self.actor_critic.policy.trainable_variables)
        )

        return {"critic_loss": critic_loss, "actor_loss": actor_loss}, td_error

    @tf.function
    def _update_target(self):
        polyak_update(
            self.actor_critic.variables, self.target_actor_critic.variables, self.tau
        )

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        cur_noise_scale = self.noise_schedule(update)

        episodes, ep_infos = self.runner.run(
            1,
            # In the beginning, randomly select actions from a uniform distribution
            # for better exploration.
            uniform_sample=(update * self.timesteps_per_iteration <= self.random_steps),
            noise_scale=cur_noise_scale,
        )

        self.replay_buffer.add(episodes.to_sample_batch())

        metrics = dict()
        if (
            update * self.timesteps_per_iteration > self.learning_starts
            and update % self.train_freq == 0
        ):
            for _ in range(self.train_freq):
                if self.prioritized_replay:
                    sample = self.replay_buffer.sample(
                        self.batch_size, self.beta_schedule(update)
                    )
                    weights = sample[SampleBatch.PRIO_WEIGHTS]
                else:
                    sample = self.replay_buffer.sample(self.batch_size)
                    weights = 1.0

                batch_metrics, td_errors = self._update(
                    sample[SampleBatch.OBS],
                    sample[SampleBatch.ACTIONS],
                    sample[SampleBatch.REWARDS],
                    sample[SampleBatch.DONES],
                    sample[SampleBatch.NEXT_OBS],
                    weights,
                    update % self.policy_delay == 0 or not self._has_updated,
                )

                self._has_updated = True

                if self.prioritized_replay:
                    self.replay_buffer.update_priorities(
                        sample["batch_indices"], td_errors
                    )

            metrics.update(batch_metrics)

            if self.num_workers != 1:
                self.runner.update_policies(self.policy.get_weights())

        if update % self.target_update_interval == 0:
            self._update_target()

        metrics["noise_scale"] = cur_noise_scale

        return metrics, ep_infos
