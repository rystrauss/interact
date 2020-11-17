from typing import Tuple, Dict, List, Union, Callable

import gin
import gym
import numpy as np
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.sac.policy import SACPolicy
from interact.agents.utils import register
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.replay_buffer import ReplayBuffer
from interact.typing import TensorType


@gin.configurable('sac', blacklist=['env_fn'])
@register('sac')
class SACAgent(Agent):
    # TODO: Add prioritized replay buffer.
    # TODO: Implement discrete action version.
    # TODO: Make sure things work if specific target_entropy is provided.

    def __init__(self,
                 env_fn: Callable[[], gym.Env],
                 network: str = 'mlp',
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 entropy_lr: float = 3e-4,
                 learning_starts: int = 1500,
                 tau: float = 5e-3,
                 initial_alpha: float = 1.0,
                 target_entropy: Union[str, float] = 'auto',
                 gamma: float = 0.99,
                 buffer_size: int = 50000,
                 train_freq: int = 1,
                 batch_size: int = 256,
                 num_workers: int = 1,
                 num_envs_per_worker: int = 1):
        super().__init__(env_fn)

        env = self.make_env()

        assert isinstance(env.action_space, gym.spaces.Box), 'Only continuous action spaces are supported currently.'

        if isinstance(target_entropy, str):
            if isinstance(env.action_space, gym.spaces.Discrete):
                target_entropy = -env.action_space.n
            else:
                target_entropy = -np.prod(env.action_space.shape)

        self.entropy_lr = entropy_lr
        self.learning_starts = learning_starts
        self.tau = tau
        self.initial_alpha = initial_alpha
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_envs_per_worker = num_envs_per_worker

        self.buffer = ReplayBuffer(buffer_size)

        def policy_fn():
            return SACPolicy(env.observation_space,
                             env.action_space,
                             network)

        def runner_policy_fn():
            return SACPolicy(env.observation_space,
                             env.action_space,
                             network,
                             actor_only=True)

        self.policy = policy_fn()
        self.target_policy = policy_fn()

        self.target_policy.trainable = False

        self.pi_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.q_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.alpha_optimizer = tf.keras.optimizers.Adam(entropy_lr)

        self.runner = Runner(env_fn, runner_policy_fn, num_envs_per_worker, num_workers)

        self.log_alpha = tf.Variable(np.log(initial_alpha), trainable=True, dtype=tf.float32)

    @property
    def timesteps_per_iteration(self) -> int:
        return self.num_workers * self.num_envs_per_worker

    @tf.function
    def act(self, obs: TensorType, state: List[TensorType] = None) -> TensorType:
        return self.policy.act(obs)

    @tf.function
    def _update_critic(self, obs, actions, rewards, next_obs, dones):
        next_actions, next_logpacs, _ = self.policy.pi(next_obs)

        target_q_values_1 = self.target_policy.q1([next_obs, next_actions])
        target_q_values_2 = self.target_policy.q2([next_obs, next_actions])

        target_q_values = tf.minimum(target_q_values_1, target_q_values_2)

        q_targets = rewards + self.gamma * (1.0 - dones) * (target_q_values - tf.exp(self.log_alpha) * next_logpacs)

        with tf.GradientTape() as tape:
            q_values_1 = self.policy.q1([obs, actions])
            q_values_2 = self.policy.q2([obs, actions])

            loss = tf.reduce_mean((q_values_1 - q_targets) ** 2) + tf.reduce_mean((q_values_2 - q_targets) ** 2)

        grads = tape.gradient(loss, self.policy.q_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.policy.q_variables))

        return {
            'critic_loss': loss
        }

    @tf.function
    def _update_policy_and_alpha(self, obs):
        with tf.GradientTape() as pi_tape, tf.GradientTape() as alpha_tape:
            actions, logpacs, entropy = self.policy.pi(obs)

            q_values_1 = self.policy.q1([obs, actions])
            q_values_2 = self.policy.q2([obs, actions])
            q_values = tf.minimum(q_values_1, q_values_2)

            pi_loss = tf.reduce_mean((tf.stop_gradient(tf.exp(self.log_alpha)) * logpacs - tf.stop_gradient(q_values)))

            alpha_loss = -tf.reduce_mean(self.log_alpha * tf.stop_gradient(logpacs + self.target_entropy))

        pi_grads = pi_tape.gradient(pi_loss, self.policy.pi.trainable_weights)
        self.pi_optimizer.apply_gradients(zip(pi_grads, self.policy.pi.trainable_weights))

        alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients((zip(alpha_grads, [self.log_alpha])))

        return {
            'policy_loss': pi_loss,
            'policy_entropy': tf.reduce_mean(entropy),
            'alpha_loss': alpha_loss,
            'alpha': tf.exp(self.log_alpha)
        }

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        episodes, ep_infos = self.runner.run(1)

        self.buffer.add(episodes.to_sample_batch())

        metrics = {}
        if update * self.timesteps_per_iteration > self.learning_starts and update % self.train_freq == 0:
            sample = self.buffer.sample(self.batch_size)

            metrics.update(self._update_critic(sample[SampleBatch.OBS],
                                               sample[SampleBatch.ACTIONS],
                                               sample[SampleBatch.REWARDS],
                                               sample[SampleBatch.NEXT_OBS],
                                               sample[SampleBatch.DONES]))

            metrics.update(self._update_policy_and_alpha(sample[SampleBatch.OBS]))

            for target_var, q_var in zip(self.target_policy.q_variables, self.policy.q_variables):
                target_var.assign(self.tau * target_var + (1 - self.tau) * q_var)

            self.runner.update_policies(self.policy.pi.get_weights())

        return metrics, ep_infos
