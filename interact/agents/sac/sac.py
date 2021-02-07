from typing import Tuple, Dict, List, Union, Callable

import gin
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from interact.agents.base import Agent
from interact.agents.sac.policy import SACPolicy, TwinQNetwork
from interact.agents.utils import register
from interact.environments.wrappers import NormalizedActionsWrapper
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.replay_buffer import ReplayBuffer
from interact.typing import TensorType


@gin.configurable("sac", blacklist=["env_fn"])
@register("sac")
class SACAgent(Agent):
    """The Soft Actor-Critic algorithm.

    This is the more modern version of the algorithm, based on:
    https://arxiv.org/abs/1812.05905

    TODO: Add support for prioritized experience replay.

    Args:
        env_fn: A function that, when called, returns an instance of the agent's environment.
        network: Base network type to be used by the policy and Q-functions.
        actor_lr: Learning rate to use for updating the actor.
        critic_lr: Learning rate to use for updating the critics.
        entropy_lr: Learning rate to use for tuning the entropy parameter.
        learning_starts: Number of timesteps to only collect experience before learning starts.
        tau: Parameter for the polyak averaging used to update the target networks.
        target_update_interval: Frequency with which the target Q-networks are updated.
        initial_alpha: The initial value of the entropy parameter.
        learn_alpha: Whether or not the alpha parameter is learned during training.
        target_entropy: The target entropy parameter. If 'auto', this is set to -|A|
            (i.e. the negative cardinality of the action set).
        gamma: The discount factor.
        buffer_size: The maximum size of the replay buffer.
        train_freq: The frequency with which training updates are performed.
        batch_size: The size of batches sampled from the replay buffer over which updates are performed.
        num_workers: The number of parallel workers to use for experience collection.
        num_envs_per_worker: The number of synchronous environments to be executed in each worker.
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
        gamma: float = 0.99,
        buffer_size: int = 50000,
        train_freq: int = 1,
        batch_size: int = 256,
        num_workers: int = 1,
        num_envs_per_worker: int = 1,
    ):
        def normalized_env_fn():
            return NormalizedActionsWrapper(env_fn())

        super().__init__(normalized_env_fn)

        env = self.make_env()

        self._discrete = isinstance(env.action_space, gym.spaces.Discrete)

        if isinstance(target_entropy, str):
            if self._discrete:
                target_entropy = 0.98 * np.array(
                    -np.log(1.0 / env.action_space.n), dtype=np.float32
                )
            else:
                target_entropy = -np.prod(env.action_space.shape)

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

        def policy_fn():
            if num_workers == 1:
                return self.policy

            return SACPolicy(env.observation_space, env.action_space, network)

        self.q_network = TwinQNetwork(env.observation_space, env.action_space, network)
        self.target_q_network = TwinQNetwork(
            env.observation_space, env.action_space, network
        )
        self.target_q_network.trainable = False

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizers = (
            tf.keras.optimizers.Adam(critic_lr),
            tf.keras.optimizers.Adam(critic_lr),
        )
        self.alpha_optimizer = (
            None if not learn_alpha else tf.keras.optimizers.Adam(entropy_lr)
        )

        self.runner = Runner(
            normalized_env_fn, policy_fn, num_envs_per_worker, num_workers
        )

        self.log_alpha = tf.Variable(
            np.log(initial_alpha), trainable=learn_alpha, dtype=tf.float32
        )
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.exp)

    @property
    def timesteps_per_iteration(self) -> int:
        return self.num_workers * self.num_envs_per_worker

    @tf.function
    def act(self, obs: TensorType, state: List[TensorType] = None) -> TensorType:
        return self.policy.act(obs)

    @tf.function
    def _update_critic(self, obs, actions, rewards, next_obs, dones):
        next_actions, next_logpacs, logits = self.policy(next_obs)

        if self._discrete:
            target_q_values = self.target_q_network(next_obs)
            target_q_values = tf.reduce_sum(tf.exp(logits) * target_q_values, axis=-1)
        else:
            target_q_values = self.target_q_network([next_obs, next_actions])

        q_targets = rewards + self.gamma * (1.0 - dones) * (
            target_q_values - self.alpha * next_logpacs
        )

        qs_values = []
        qs_losses = []

        for q_net, optimizer in zip(
            (self.q_network.q1, self.q_network.q2), self.q_optimizers
        ):
            with tf.GradientTape() as tape:
                if self._discrete:
                    q_values = q_net(obs)
                    num_actions = q_values.shape[-1]
                    q_values = tf.reduce_sum(
                        q_values * tf.one_hot(next_actions, num_actions), axis=-1
                    )
                else:
                    q_values = q_net([obs, actions])

                loss = 0.5 * tf.losses.mse(
                    q_targets[:, tf.newaxis], q_values[:, tf.newaxis]
                )
                loss = tf.nn.compute_average_loss(loss)

            grads = tape.gradient(loss, q_net.trainable_variables)
            optimizer.apply_gradients(zip(grads, q_net.trainable_variables))

            qs_losses.append(loss)
            qs_values.append(q_values)

        return {
            "critic_loss": tf.reduce_mean(qs_losses),
            "q_values_mean": tf.reduce_mean(q_values),
        }

    @tf.function
    def _update_policy_and_alpha(self, obs):
        with tf.GradientTape() as actor_tape, tf.GradientTape() as alpha_tape:
            actions, logpacs, logits = self.policy(obs)
            probs = tf.exp(logits)

            if self._discrete:
                q_values = self.q_network(obs)

                actor_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.multiply(
                            probs,
                            tf.stop_gradient(self.alpha) * logits
                            - tf.stop_gradient(q_values),
                        ),
                        axis=-1,
                    )
                )

                if self.learn_alpha:
                    alpha_loss = -tf.reduce_mean(
                        tf.reduce_sum(
                            tf.multiply(
                                tf.stop_gradient(probs),
                                self.log_alpha
                                * tf.stop_gradient(logits + self.target_entropy),
                            ),
                            axis=-1,
                        )
                    )
                else:
                    alpha_loss = 0
            else:
                q_values = self.q_network([obs, actions])

                actor_loss = self.alpha * logpacs - q_values
                actor_loss = tf.nn.compute_average_loss(actor_loss)

                if self.learn_alpha:
                    alpha_loss = -1.0 * (
                        self.alpha * tf.stop_gradient(logpacs + self.target_entropy)
                    )
                    alpha_loss = tf.nn.compute_average_loss(alpha_loss)
                else:
                    alpha_loss = 0

        actor_grads = actor_tape.gradient(actor_loss, self.policy.trainable_weights)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.policy.trainable_weights)
        )

        if self.learn_alpha:
            alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients((zip(alpha_grads, [self.log_alpha])))

        return {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": self.alpha,
        }

    @tf.function
    def _update_target(self):
        for target_var, q_var in zip(
            self.target_q_network.variables, self.q_network.variables
        ):
            target_var.assign(self.tau * target_var + (1 - self.tau) * q_var)

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        episodes, ep_infos = self.runner.run(1)

        self.buffer.add(episodes.to_sample_batch())

        metrics = dict()
        if (
            update * self.timesteps_per_iteration > self.learning_starts
            and update % self.train_freq == 0
        ):
            sample = self.buffer.sample(self.batch_size)

            metrics.update(
                self._update_critic(
                    sample[SampleBatch.OBS],
                    sample[SampleBatch.ACTIONS],
                    sample[SampleBatch.REWARDS],
                    sample[SampleBatch.NEXT_OBS],
                    sample[SampleBatch.DONES],
                )
            )

            # metrics.update(self._update_policy_and_alpha(sample[SampleBatch.OBS]))

            if update % self.target_update_interval == 0:
                self._update_target()

            if self.num_workers != 1:
                self.runner.update_policies(self.policy.get_weights())

        return metrics, ep_infos
