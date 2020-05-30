"""Implementation of the deep q-learning algorithm.

Author: Ryan Strauss
"""

import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from interact.agents import Agent
from interact.agents.deepq.policy import DeepQPolicy
from interact.agents.deepq.replay_buffer import ReplayBuffer
from interact.agents.util import register
from interact.common.math_util import safe_mean
from interact.common.networks import build_network_fn
from interact.common.schedules import LinearDecay


@register('deepq')
class DQNAgent(Agent):
    """

    Args:
        env: The environment the agent is interacting with.
        load_path: A path to a checkpoint that will be loaded before training begins. If None, agent parameters
            will be initialized from scratch.
        feature_extraction: The type of network to be used for feature extraction portion of the Q-networks.
        batch_size: The size of experience batches sampled from the replay buffer.
        buffer_size: The size of the experience replay buffer. Network update are sampled from this
            number of most recent frames.
        train_freq: The frequency (measured in the number of updates) with which the online Q- network is updated.
        target_update_freq: The frequency (measured in the number of updates) with which the target
            network is updated.
        gamma: Discount factor.
        learning_rate: The learning rate.
        initial_exploration: Initial value of epsilon in e-greedy exploration.
        final_exploration: Final value of epsilon in e-greedy exploration.
        exploration_fraction: The percentage of the training period over which epsilon is annealed.
        learning_starts (int): A uniform random policy is run for this number of steps before learning starts,
            and the resulting experience is used to populate the replay memory.
        double: If True, Double Q-Learning is used.
        **network_kwargs: Keyword arguments to be passed to the networks.
    """

    def __init__(self, *, env, load_path=None, feature_extraction='cnn', batch_size=32, buffer_size=50000, train_freq=1,
                 target_update_freq=500, gamma=0.99, learning_rate=5e-4, initial_exploration=1., final_exploration=0.02,
                 exploration_fraction=0.1, learning_starts=1000, double=False, **network_kwargs):
        assert env.num_envs == 1, 'DQNAgent cannot use parallelized environments -- `num_envs` should be set to 1'

        self.policy = DeepQPolicy(
            env.action_space, build_network_fn(feature_extraction, env.observation_space.shape, **network_kwargs))

        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.gamma = gamma
        self.initial_exploration = initial_exploration
        self.final_exploration = final_exploration
        self.exploration_fraction = exploration_fraction
        self.learning_starts = learning_starts
        self.double = double

        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        super().__init__(env=env, load_path=load_path)

    @tf.function
    def _train_step(self, obs, actions, rewards, next_obs, dones):
        """Performs a gradient descent update for the online Q network.

        Args:
            obs: Observations from a sample from the experience replay buffer.
            actions: The actions taken after each of the corresponding observations from the experience replay buffer.
            rewards: The rewards garnered after taking the respective actions.
            next_obs: The new observations from the environment (at the next time step) after taking the
                actions from a sample from the experience replay buffer.
            dones: A boolean array where true indicates that a respective experience was terminal.

        Returns:
            None.
        """
        dones = tf.cast(dones, tf.float32)
        num_actions = self.env.action_space.n

        # Get Q values for the observations at the next time step with the target network
        next_q_values = self.policy.target_network(next_obs)

        if self.double:
            # If using double DQN, we use online network to select actions
            selected_actions = tf.argmax(self.policy(next_obs), -1)
            # The one-hot matrix allows us to extract only the Q-values for our selected actions
            next_q_values_best = tf.reduce_sum(next_q_values * tf.one_hot(selected_actions, num_actions), -1)
        else:
            # In vanilla DQN, simply use the maximum value
            next_q_values_best = tf.reduce_max(next_q_values, -1)

        # Set Q values for terminal experiences to zero
        next_q_values_best_masked = (1. - dones) * next_q_values_best
        # Calculate targets (Bellman equation)
        targets = rewards + self.gamma * next_q_values_best_masked

        with tf.GradientTape() as tape:
            # Forward pass through online Q network
            q_values = self.policy(obs)

            # Q scores for actions which we know were selected in each state; we again use the one-hot trick as above
            q_values_selected = tf.reduce_sum(q_values * tf.one_hot(actions, num_actions), -1)

            # Compute loss; Huber loss implements the gradient clipping described by Mnih et. al.
            # Note that the meaning of the gradient clipping they describe can be easily misinterpreted
            # as saying to clip the objective. It is really saying to clip the multiplicative term when computing
            # the gradient, which is what the Huber loss does for us.
            loss = tf.keras.losses.Huber()(targets, q_values_selected)

        # Calculate gradients of the loss with respect to the network parameters
        vars = self.policy.q_network.trainable_variables
        grads = tape.gradient(loss, vars)
        # Apply the gradient update
        self.optimizer.apply_gradients(zip(grads, vars))

    def learn(self, *, total_timesteps, logger, log_interval=100, save_interval=None):
        epsilon = LinearDecay(initial_learning_rate=self.initial_exploration,
                              decay_steps=total_timesteps * self.exploration_fraction,
                              end_learning_rate=self.final_exploration)

        replay_buffer = ReplayBuffer(self.buffer_size)

        self.policy.update_target()

        episode_rewards = [0.]

        obs = self.env.reset()

        for t in tqdm(range(1, total_timesteps + 1), desc='Updates'):
            # First we select an action
            if np.random.rand() < epsilon(t):
                # With some probability epsilon, we take a random action
                action = self.env.action_space.sample()
            else:
                # Otherwise, take the action recommended by the policy
                action = self.policy.step(obs)[0][0].numpy()

            # Take a step in the environment with the chosen action
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to the replay buffer
            replay_buffer.add(obs[0], action, reward[0], next_obs[0], done[0])
            obs = next_obs
            episode_rewards[-1] += reward

            if done:
                episode_rewards.append(0.)

            # Sample a batch from the experience replay and use it for the Q network update
            if t > self.learning_starts and t % self.train_freq == 0:
                sample = replay_buffer.sample(self.batch_size)
                self._train_step(*sample)

            # Periodically update target network
            if t > self.learning_starts and t % self.target_update_freq == 0:
                self.policy.update_target()

            # Periodically log to TensorBoard
            if t % log_interval == 0 or t == 1:
                logger.log_scalar(t, 'total_timesteps', t)
                logger.log_scalar(t, 'epsilon', epsilon(t))
                logger.log_scalar(t, 'episode/num_episodes', len(episode_rewards))
                logger.log_scalar(t, 'episode/reward_mean', safe_mean(episode_rewards[-101:-1]))

            # Periodically save model weights
            if (save_interval is not None and t % save_interval == 0) or t == total_timesteps:
                self.save(os.path.join(logger.directory, 'weights', f'update_{t}'))

    def act(self, observation):
        return self.policy.step(observation)[0]

    def load(self, path):
        self.policy.q_network.load_weights(path)
        self.policy.update_target()

    def save(self, path):
        self.policy.q_network.save_weights(path)
