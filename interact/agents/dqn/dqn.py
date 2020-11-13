from typing import Tuple, Dict, List, Callable

import gin
import gym
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.dqn.policy import DQNPolicy
from interact.agents.utils import register
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.replay_buffer import ReplayBuffer
from interact.schedules import LinearDecay
from interact.typing import TensorType


@gin.configurable(name_or_fn='dqn', blacklist=['env_fn'])
@register('dqn')
class DQNAgent(Agent):

    def __init__(self,
                 env_fn: Callable[[], gym.Env],
                 q_network: str = 'mlp',
                 batch_size: int = 32,
                 buffer_size: int = 50000,
                 train_freq: int = 1,
                 target_update_freq: int = 500,
                 gamma: float = 0.99,
                 lr: float = 5e-4,
                 initial_epsilon: float = 1.0,
                 final_epsilon: float = 0.02,
                 epsilon_timesteps: int = 10000,
                 learning_starts: int = 1000,
                 max_grad_norm: float = None,
                 double: bool = True):
        super().__init__(env_fn)

        self.batch_size = batch_size
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_timesteps = epsilon_timesteps
        self.learning_starts = learning_starts
        self.max_grad_norm = max_grad_norm
        self.double = double

        def policy_fn():
            return DQNPolicy(self.env.observation_space, self.env.action_space, q_network)

        self.env = self.make_env()
        self.policy = policy_fn()
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.epsilon = None
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.runner = Runner(env_fn, policy_fn, 1, 1)

    @property
    def timesteps_per_iteration(self) -> int:
        return 1

    @tf.function
    def _update(self, obs, actions, rewards, next_obs, dones):
        dones = tf.cast(tf.squeeze(dones), tf.float32)
        actions = tf.cast(tf.squeeze(actions), tf.int32)
        rewards = tf.squeeze(rewards)
        num_actions = self.env.action_space.n

        # Get Q values for the observations at the next time step with the target network
        next_q_values = self.policy.target_network(next_obs)

        if self.double:
            # If using double DQN, we use online network to select actions
            selected_actions = tf.argmax(self.policy(next_obs), axis=-1)
            # The one-hot matrix allows us to extract only the Q-values for our selected actions
            next_q_values_best = tf.reduce_sum(next_q_values * tf.one_hot(selected_actions, num_actions), axis=-1)
        else:
            # In vanilla DQN, simply use the maximum value
            next_q_values_best = tf.reduce_max(next_q_values, axis=-1)

        # Set Q values for terminal experiences to zero
        next_q_values_best_masked = (1. - dones) * next_q_values_best
        # Calculate targets (Bellman equation)
        targets = rewards + self.gamma * next_q_values_best_masked

        with tf.GradientTape() as tape:
            # Forward pass through online Q network
            q_values = self.policy(obs)

            # Q scores for actions which we know were selected in each state; we again use the one-hot trick as above
            q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), axis=-1)

            # Compute loss; Huber loss implements the gradient clipping described by Mnih et. al.
            # Note that the meaning of the gradient clipping they describe can be easily misinterpreted
            # as saying to clip the objective. It is really saying to clip the multiplicative term when computing
            # the gradient, which is what the Huber loss does for us.
            targets = tf.expand_dims(targets, axis=1)
            q_values_selected = tf.expand_dims(q_values_selected, axis=1)
            loss = tf.keras.losses.Huber()(targets, q_values_selected)

        # Calculate gradients of the loss with respect to the network parameters
        vars = self.policy.q_network.trainable_variables
        grads = tape.gradient(loss, vars)
        # Perform gradient clipping
        if self.max_grad_norm:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.optimizer.apply_gradients(zip(grads, vars))

    @tf.function
    def act(self, obs: TensorType, state: List[TensorType] = None) -> TensorType:
        q_values = self.policy(obs)
        return tf.argmax(q_values, axis=-1).numpy()

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        self.runner.update_policies(self.policy.get_weights())
        batch, ep_infos = self.runner.run(1, epsilon=self.epsilon(update))

        self.replay_buffer.add(batch.to_sample_batch())

        if update > self.learning_starts and update % self.train_freq == 0:
            sample = self.replay_buffer.sample(self.batch_size)
            self._update(sample[SampleBatch.OBS],
                         sample[SampleBatch.ACTIONS],
                         sample[SampleBatch.REWARDS],
                         sample[SampleBatch.NEXT_OBS],
                         sample[SampleBatch.DONES])

        # Periodically update target network
        if update > self.learning_starts and update % self.target_update_freq == 0:
            self.policy.update_target_network()

        info = {
            'epsilon': self.epsilon(update)
        }

        return info, ep_infos

    def setup(self, total_timesteps: int):
        self.epsilon = LinearDecay(initial_learning_rate=self.initial_epsilon,
                                   decay_steps=self.epsilon_timesteps,
                                   end_learning_rate=self.final_epsilon)
