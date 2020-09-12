"""Implementation of the Phasic Policy Gradient algorithm.

Author: Ryan Strauss
"""

import os

import numpy as np
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.ppg.policy import PPGPolicy
from interact.agents.ppo.runner import Runner
from interact.agents.util import register
from interact.common.networks import build_network_fn
from interact.logger import Logger


@register('ppg')
class PPGAgent(Agent):
    """An agent that learns using the Phasic Policy Gradient algorithm.

    Args:
        env: The environment the agent is interacting with.
        load_path: A path to a checkpoint that will be loaded before training begins. If None, agent parameters
            will be initialized from scratch.
        policy_network: The type of network to be used for the policy.
        value_network: The method of constructing the value network, either 'shared' or 'copy'.
        gamma: The discount factor.
        nsteps: The number of steps taken in each environment per update.
        ent_coef: The coefficient of the entropy term in the loss function.
        vf_coef: The coefficiant of the value term in the loss function.
        learning_rate: The initial learning rate.
        lr_schedule: The schedule for the learning rate, either 'constant' or 'linear'.
        max_grad_norm: The maximum value for the gradient clipping.
        lam: Lambda value used in the GAE calculation.
        nminibatches: Number of training minibatches per update.
        noptepochs: Number of epochs over each batch when optimizing the loss.
        cliprange: Clipping parameter used in the surrogate loss.
        cliprange_schedule: The schedule for the clipping parameter, either 'constant' or 'linear'.
        **network_kwargs: Keyword arguments to be passed to the policy/value network.
    """

    def __init__(self, *, env, load_path=None, network='mlp', gamma=0.99, nsteps=2048,
                 ent_coef=0.0, vf_coef=0.5, learning_rate=3e-4, max_grad_norm=0.5, lam=0.95,
                 nminibatches=4, noptepochs=4, cliprange=0.2, policy_iterations=32,
                 policy_epochs=1, value_epochs=1, auxiliary_epochs=6, bc_coef=1.0, nminibatches_aux=16,
                 **network_kwargs):
        self.policy = PPGPolicy(env.action_space,
                                build_network_fn(network, env.observation_space.shape, **network_kwargs))
        self.gamma = gamma
        self.nsteps = nsteps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.cliprange = cliprange
        self.policy_iterations = policy_iterations
        self.policy_epochs = policy_epochs
        self.value_epochs = value_epochs
        self.auxiliary_epochs = auxiliary_epochs
        self.bc_coef = bc_coef
        self.nminibatches_aux = nminibatches_aux

        self.policy_optimizer = None
        self.value_optimizer = None
        self.aux_optimizer = None

        super().__init__(env=env, load_path=load_path)

    def _compute_losses(self, rollout, compute_policy_loss=True, compute_value_loss=True):
        obs, returns, actions, values, neglogpacs_old = rollout

        advantages = returns - values
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        advantages = tf.stop_gradient(advantages)

        if compute_policy_loss:
            # Compute the policy for the given observations
            pi = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = tf.reduce_sum(tf.reshape(-pi.log_prob(actions), (len(actions), -1)), axis=-1)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the policy surrogate loss
            ratio = tf.exp(neglogpacs_old - neglogpacs)
            pg_loss_unclipped = -advantages * ratio
            pg_loss_clipped = -advantages * tf.clip_by_value(ratio, 1 - self.cliprange, 1 + self.cliprange)
            policy_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
            policy_loss = policy_loss - entropy * self.ent_coef

            if not compute_value_loss:
                return policy_loss

        if compute_value_loss:
            value_preds = self.policy.value(obs)
            value_preds_clipped = tf.clip_by_value(value_preds, -self.cliprange, self.cliprange)
            vf_loss_unclipped = (returns - value_preds) ** 2
            vf_loss_clipped = (returns - value_preds_clipped) ** 2
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss_clipped, vf_loss_unclipped))

            if not compute_policy_loss:
                return value_loss

        return policy_loss, value_loss

    @tf.function
    def _train_policy(self, rollout):
        with tf.GradientTape() as tape:
            policy_loss = self._compute_losses(rollout, compute_value_loss=False)

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(policy_loss, self.policy.trainable_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        return policy_loss

    @tf.function
    def _train_value(self, rollout):
        with tf.GradientTape() as tape:
            value_loss = self._compute_losses(rollout, compute_policy_loss=False)
            loss = value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.trainable_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.value_optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        return value_loss

    @tf.function
    def _train_policy_and_value(self, rollout):
        with tf.GradientTape() as tape:
            policy_loss, value_loss = self._compute_losses(rollout)
            loss = policy_loss + value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.trainable_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        return policy_loss, value_loss

    @tf.function
    def _train_auxiliary(self, obs, returns, pis):
        with tf.GradientTape() as tape:
            # TODO: Finish this -- add L^joint

            value_preds = self.policy.value(obs)
            value_preds_clipped = tf.clip_by_value(value_preds, -self.cliprange, self.cliprange)
            vf_loss_unclipped = (returns - value_preds) ** 2
            vf_loss_clipped = (returns - value_preds_clipped) ** 2
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss_clipped, vf_loss_unclipped))

    def _minibatch_optimize(self, update_fn, rollout, batch_size, epochs, nminibatches):
        mb_losses = []
        indices = np.arange(batch_size)
        minibatch_size = batch_size // nminibatches

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                rollout_slices = (x[mb_indices] for x in (*rollout,))
                mb_losses.append(update_fn(*rollout_slices))

        return mb_losses

    def _policy_phase(self, runner, total_timesteps):
        obs_buffer = []
        returns_buffer = []

        for _ in range(self.policy_iterations):
            if runner.steps >= total_timesteps:
                break

            *rollout, ep_infos = runner.run()

            if self.policy_epochs == self.value_epochs:
                losses = self._minibatch_optimize(self._train_policy_and_value, rollout, runner.batch_size,
                                                  self.policy_epochs,
                                                  self.nminibatches)
            else:
                policy_losses = self._minibatch_optimize(self._train_policy, rollout, runner.batch_size,
                                                         self.policy_epochs,
                                                         self.nminibatches)

                value_losses = self._minibatch_optimize(self._train_value, rollout, runner.batch_size,
                                                        self.value_epochs,
                                                        self.nminibatches)

                losses = list(zip(policy_losses, value_losses))

            # TODO: do logging here

            obs_buffer.append(rollout[0])
            returns_buffer.append(rollout[1])

        return obs_buffer, returns_buffer, losses

    def _auxiliary_phase(self, obs_buffer, returns_buffer, pi_buffer):
        batch_size = len(obs_buffer)
        indices = np.arange(batch_size)
        minibatch_size = batch_size // self.nminibatches_aux

        buffer_data = obs_buffer, returns_buffer, pi_buffer
        mb_losses = []

        for _ in range(self.auxiliary_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_indices = indices[start:end]
                buffer_slices = (x[mb_indices] for x in (*buffer_data,))
                mb_losses.append(self._train_auxiliary(*buffer_slices))

        return mb_losses

    def learn(self, *, total_timesteps, logger, log_interval=100, save_interval=None):
        assert isinstance(logger, Logger), 'logger must be an instance of the `Logger` class'

        # Create the runner that collects experience
        runner = Runner(self.env, self.policy, self.nsteps, self.gamma, self.lam)

        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        if self.policy_epochs != self.value_epochs:
            self.value_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.aux_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        while True:
            if runner.steps >= total_timesteps:
                break

            obs_buffer, returns_buffer, losses = self._policy_phase(runner, total_timesteps)

            pi_buffer = [self.policy(obs) for obs in obs_buffer]
            obs_buffer = np.vstack(obs_buffer)
            returns_buffer = np.vstack(returns_buffer)

            # Periodically save model weights
            if (save_interval is not None and update % save_interval == 0) or update == nupdates:
                self.save(os.path.join(logger.directory, 'weights', f'update_{update}'))

    @tf.function
    def act(self, observation):
        pi = self.policy(observation)
        return pi.mode()

    def load(self, path):
        self.policy.load_weights(path)

    def save(self, path):
        self.policy.save_weights(path)
