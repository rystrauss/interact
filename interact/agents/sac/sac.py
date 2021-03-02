from typing import Tuple, Dict, List, Union, Callable

import gin
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from interact.agents.base import Agent
from interact.agents.sac.policy import SACPolicy, TwinQNetwork
from interact.agents.utils import register
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.replay_buffer import ReplayBuffer
from interact.typing import TensorType


@gin.configurable("sac", denylist=["env_fn"])
@register("sac")
class SACAgent(Agent):
    """The Soft Actor-Critic algorithm.

    This is the more modern version of the algorithm, based on:
    https://arxiv.org/abs/1812.05905

    TODO: Add support for prioritized experience replay.

    Args:
        env_fn: A function that, when called, returns an instance of the agent's
            environment.
        network: Base network type to be used by the policy and Q-functions.
        actor_lr: Learning rate to use for updating the actor.
        critic_lr: Learning rate to use for updating the critics.
        entropy_lr: Learning rate to use for tuning the entropy parameter.
        learning_starts: Number of timesteps to only collect experience before learning
            starts.
        tau: Parameter for the polyak averaging used to update the target networks.
        target_update_interval: Frequency with which the target Q-networks are updated.
        initial_alpha: The initial value of the entropy parameter.
        learn_alpha: Whether or not the alpha parameter is learned during training.
        target_entropy: The target entropy parameter. If 'auto', this is set to -|A|
            (i.e. the negative cardinality of the action set).
        gamma: The discount factor.
        buffer_size: The maximum size of the replay buffer.
        train_freq: The frequency with which training updates are performed.
        batch_size: The size of batches sampled from the replay buffer over which
            updates are performed.
        num_workers: The number of parallel workers to use for experience collection.
        num_envs_per_worker: The number of synchronous environments to be executed in
            each worker.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        network: str = "mlp",
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        entropy_lr: float = 3e-4,
        learning_starts: int = 1500,
        tau: float = 5e-3,
        target_update_interval: int = 1,
        initial_alpha: float = 1.0,
        learn_alpha: bool = True,
        target_entropy: Union[str, float] = "auto",
        gamma: float = 0.95,
        buffer_size: int = 50000,
        train_freq: int = 1,
        batch_size: int = 256,
        num_workers: int = 1,
        num_envs_per_worker: int = 1,
    ):
        super().__init__(env_fn)

        env = self.make_env()

        self._discrete = isinstance(env.action_space, gym.spaces.Discrete)

        if isinstance(target_entropy, str):
            if self._discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / env.action_space.n), dtype=np.float32
                )
            else:
                target_entropy = -np.prod(env.action_space.shape)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.entropy_lr = entropy_lr
        self.learning_starts = learning_starts
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.initial_alpha = initial_alpha
        self.learn_alpha = learn_alpha
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker

        self.buffer = ReplayBuffer(buffer_size)

        self.policy = SACPolicy(env.observation_space, env.action_space, network)
        self.policy.build([None, *env.observation_space.shape])

        def policy_fn():
            if num_workers == 1:
                return self.policy

            return SACPolicy(env.observation_space, env.action_space, network)

        self.q_network = TwinQNetwork(env.observation_space, env.action_space, network)
        self.target_q_network = TwinQNetwork(
            env.observation_space, env.action_space, network
        )
        self.target_q_network.trainable = False

        if (
            not isinstance(env.action_space, gym.spaces.Discrete)
            and len(env.observation_space.shape) == 1
        ):
            q_input_shape = (
                env.observation_space.shape[0] + env.action_space.shape[0],
            )
        else:
            q_input_shape = env.observation_space.shape

        q_input_shape = (None, *q_input_shape)

        self.q_network.build(q_input_shape)
        self.target_q_network.build(q_input_shape)
        self.target_q_network.set_weights(self.q_network.get_weights())

        self.actor_optimizer = None
        self.critic_optimizer = None
        self.alpha_optimizer = None

        self.runner = None
        self.runner_config = dict(
            env_fn=env_fn,
            policy_fn=policy_fn,
            num_envs_per_worker=num_envs_per_worker,
            num_workers=num_workers,
        )

        self.log_alpha = tf.Variable(
            np.log(initial_alpha), trainable=learn_alpha, dtype=tf.float32
        )
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)

    @property
    def timesteps_per_iteration(self) -> int:
        return self.num_envs_per_worker * self.num_workers

    @tf.function
    def _continuous_update(self, obs, actions, rewards, dones, next_obs):
        # UPDATE CRITIC
        next_actions, next_logpacs = self.policy(next_obs)

        q_targets = tf.minimum(*self.target_q_network([next_obs, next_actions]))
        backup = rewards + self.gamma * (1.0 - dones) * (
            q_targets - self.alpha * next_logpacs
        )

        with tf.GradientTape() as tape:
            q1_values, q2_values = self.q_network([obs, actions])

            q1_loss = tf.losses.huber(backup, q1_values)
            q2_loss = tf.losses.huber(backup, q2_values)
            critic_loss = 0.5 * (q1_loss + q2_loss)

        grads = tape.gradient(critic_loss, self.q_network.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(grads, self.q_network.trainable_variables)
        )

        # UPDATE POLICY
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy.trainable_variables)

            pi, logpacs = self.policy(obs)
            q_targets = tf.minimum(*self.q_network([obs, pi]))

            actor_loss = tf.reduce_mean(self.alpha * logpacs - q_targets)

        grads = tape.gradient(actor_loss, self.policy.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads, self.policy.trainable_variables)
        )

        # UPDATE ALPHA
        if self.learn_alpha:
            with tf.GradientTape() as tape:
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(logpacs + self.target_entropy)
                )

            grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(grads, [self.log_alpha]))
        else:
            alpha_loss = 0.0

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
        }

    @tf.function
    def _discrete_update(self, obs, actions, rewards, dones, next_obs):
        # UPDATE CRITIC
        _, next_logpacs = self.policy(next_obs)

        q_targets = tf.minimum(*self.target_q_network(next_obs))
        q_targets = tf.reduce_sum(
            tf.exp(next_logpacs) * (q_targets - self.alpha * next_logpacs), axis=-1
        )
        backup = rewards + self.gamma * (1.0 - dones) * q_targets

        with tf.GradientTape() as tape:
            q1_values, q2_values = self.q_network(obs)

            actions = tf.cast(actions, tf.int32)
            q1_values = tf.gather(q1_values, actions, batch_dims=1)
            q2_values = tf.gather(q2_values, actions, batch_dims=1)

            q1_loss = tf.losses.huber(backup, q1_values)
            q2_loss = tf.losses.huber(backup, q2_values)
            critic_loss = 0.5 * (q1_loss + q2_loss)

        grads = tape.gradient(critic_loss, self.q_network.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(grads, self.q_network.trainable_variables)
        )

        # UPDATE POLICY
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.policy.trainable_variables)

            pi, logpacs = self.policy(obs)
            q_values = tf.minimum(*self.q_network(obs))

            actor_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.multiply(
                        tf.exp(logpacs),
                        self.alpha * logpacs - tf.stop_gradient(q_values),
                    ),
                    axis=-1,
                )
            )

        grads = tape.gradient(actor_loss, self.policy.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(grads, self.policy.trainable_variables)
        )

        # UPDATE ALPHA
        if self.learn_alpha:
            with tf.GradientTape() as tape:
                alpha_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(
                            tf.exp(logpacs),
                            -self.log_alpha
                            * tf.stop_gradient(logpacs + self.target_entropy),
                        ),
                        axis=-1,
                    )
                )

            grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(grads, [self.log_alpha]))
        else:
            alpha_loss = 0.0

        return {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
        }

    @tf.function
    def _update_target(self):
        """Perform Polyak averaging of the target network."""
        for target_var, q_var in zip(
            self.target_q_network.variables, self.q_network.variables
        ):
            target_var.assign(self.tau * q_var + (1 - self.tau) * target_var)

    def _update(self, update, sample):
        metrics = dict()
        if self._discrete:
            metrics.update(
                self._discrete_update(
                    sample[SampleBatch.OBS],
                    sample[SampleBatch.ACTIONS],
                    sample[SampleBatch.REWARDS],
                    sample[SampleBatch.DONES],
                    sample[SampleBatch.NEXT_OBS],
                )
            )
        else:
            metrics.update(
                self._continuous_update(
                    sample[SampleBatch.OBS],
                    sample[SampleBatch.ACTIONS],
                    sample[SampleBatch.REWARDS],
                    sample[SampleBatch.DONES],
                    sample[SampleBatch.NEXT_OBS],
                )
            )

        if update % self.target_update_interval == 0:
            self._update_target()

        if self.num_workers != 1:
            self.runner.update_policies(self.policy.get_weights())

        return metrics

    @tf.function
    def act(self, obs: TensorType, deterministic: bool = True) -> TensorType:
        return self.policy(obs, deterministic=deterministic)[0]

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        episodes, ep_infos = self.runner.run(
            1,
            # In the beginning, randomly select actions from a uniform distribution
            # for better exploration.
            uniform_sample=(
                update * self.timesteps_per_iteration <= self.learning_starts
            ),
        )

        self.buffer.add(episodes.to_sample_batch())

        metrics = dict()
        if (
            update * self.timesteps_per_iteration > self.learning_starts
            and update % self.train_freq == 0
        ):
            sample = self.buffer.sample(self.batch_size)

            metrics.update(self._update(update, sample))

        return metrics, ep_infos

    def pretrain_setup(self, total_timesteps: int):
        self.runner = Runner(**self.runner_config)

        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.alpha_optimizer = (
            None if not self.learn_alpha else tf.keras.optimizers.Adam(self.entropy_lr)
        )
