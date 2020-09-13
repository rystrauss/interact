"""Implementation of the Phasic Policy Gradient algorithm.

Author: Ryan Strauss
"""
import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from interact.agents.base import Agent
from interact.agents.ppg.policy import PPGPolicy
from interact.agents.ppo.runner import Runner
from interact.agents.util import register
from interact.common.math_util import explained_variance, safe_mean
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

    def __init__(self, *, env, load_path=None, network='mlp', gamma=0.999, nsteps=256,
                 ent_coef=0.1, vf_coef=0.5, learning_rate=5e-4, max_grad_norm=0.5, lam=0.95,
                 nminibatches=8, noptepochs=4, cliprange=0.2, policy_iterations=32,
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

    def _compute_losses(self, obs, returns, actions, values, neglogpacs_old, compute_policy_loss=True,
                        compute_value_loss=True):
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

            if not compute_value_loss:
                return policy_loss, entropy

        if compute_value_loss:
            value_preds = self.policy.value(obs)
            value_loss = 0.5 * tf.reduce_mean((returns - value_preds) ** 2)

            if not compute_policy_loss:
                return value_loss

        return policy_loss, entropy, value_loss

    @tf.function
    def _train_policy(self, obs, returns, actions, values, neglogpacs_old):
        with tf.GradientTape() as tape:
            policy_loss, entropy = self._compute_losses(obs, returns, actions, values, neglogpacs_old,
                                                        compute_value_loss=False)
            loss = policy_loss - entropy * self.ent_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.policy_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.policy_weights))

        return policy_loss, entropy

    @tf.function
    def _train_value(self, obs, returns, actions, values, neglogpacs_old):
        with tf.GradientTape() as tape:
            value_loss = self._compute_losses(obs, returns, actions, values, neglogpacs_old, compute_policy_loss=False)
            loss = value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.value_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.value_optimizer.apply_gradients(zip(grads, self.policy.value_weights))

        return value_loss

    @tf.function
    def _train_policy_and_value(self, obs, returns, actions, values, neglogpacs_old):
        with tf.GradientTape() as tape:
            policy_loss, entropy, value_loss = self._compute_losses(obs, returns, actions, values, neglogpacs_old)
            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.policy_and_value_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.policy_and_value_weights))

        return policy_loss, entropy, value_loss

    @tf.function
    def _train_auxiliary(self, obs, returns, old_pi_logits):
        old_pi = self.policy.make_pdf(old_pi_logits)
        with tf.GradientTape() as tape:
            pi, aux_value_preds = self.policy.auxiliary_heads(obs)

            bc_loss = tf.reduce_mean(old_pi.kl_divergence(pi))
            aux_value_loss = 0.5 * tf.reduce_mean((returns - aux_value_preds) ** 2)

            joint_loss = aux_value_loss + self.bc_coef * bc_loss

        grads = tape.gradient(joint_loss, self.policy.auxiliary_weights)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.aux_optimizer.apply_gradients(zip(grads, self.policy.auxiliary_weights))

        with tf.GradientTape() as tape:
            value_preds = self.policy.value(obs)
            value_loss = 0.5 * tf.reduce_mean((returns - value_preds) ** 2)

        grads = tape.gradient(value_loss, self.policy.value_weights)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)

        if self.value_optimizer is not None:
            self.value_optimizer.apply_gradients(zip(grads, self.policy.value_weights))
        else:
            self.policy_optimizer.apply_gradients(zip(grads, self.policy.value_weights))

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

        return np.mean(mb_losses, axis=0).tolist()

    def _policy_phase(self, runner, total_timesteps, ep_info_buf):
        obs_buffer = []
        returns_buffer = []

        for _ in range(self.policy_iterations):
            if runner.steps >= total_timesteps:
                break

            *rollout, ep_infos = runner.run()
            ep_info_buf.extend(ep_infos)

            if self.policy_epochs == self.value_epochs:
                policy_loss, entropy, value_loss = self._minibatch_optimize(self._train_policy_and_value, rollout,
                                                                            runner.batch_size,
                                                                            self.policy_epochs,
                                                                            self.nminibatches)
            else:
                policy_loss, entropy = self._minibatch_optimize(self._train_policy, rollout, runner.batch_size,
                                                                self.policy_epochs,
                                                                self.nminibatches)

                value_loss = self._minibatch_optimize(self._train_value, rollout, runner.batch_size,
                                                      self.value_epochs,
                                                      self.nminibatches)

            obs_buffer.append(rollout[0])
            returns_buffer.append(rollout[1])

        return obs_buffer, returns_buffer, policy_loss, entropy, value_loss, explained_variance(rollout[3], rollout[1])

    def _auxiliary_phase(self, obs_buffer, returns_buffer):
        obs_buffer = np.vstack(obs_buffer)
        returns_buffer = np.array(returns_buffer).flatten()

        batch_size = len(obs_buffer)
        indices = np.arange(batch_size)
        minibatch_size = batch_size // self.nminibatches_aux

        obs_mbs = []
        returns_mbs = []
        old_pi_logit_mbs = []

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]
            obs_mb, ret_mb = (x[mb_indices] for x in (obs_buffer, returns_buffer,))
            obs_mbs.append(obs_mb)
            returns_mbs.append(ret_mb)
            old_pi_logit_mbs.append(self.policy.policy_logits(obs_mb))

        for _ in range(self.auxiliary_epochs):
            for i in np.random.permutation(np.arange(self.nminibatches_aux)):
                self._train_auxiliary(obs_mbs[i], returns_mbs[i], old_pi_logit_mbs[i])

    def learn(self, *, total_timesteps, logger, log_interval=100, save_interval=None):
        assert isinstance(logger, Logger), 'logger must be an instance of the `Logger` class'

        # Create the runner that collects experience
        runner = Runner(self.env, self.policy, self.nsteps, self.gamma, self.lam)

        self.policy_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        if self.policy_epochs != self.value_epochs:
            self.value_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.aux_optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)

        pbar = tqdm(total=total_timesteps, desc='Timesteps')

        update = 0
        ep_info_buf = []

        while True:
            if runner.steps >= total_timesteps:
                break

            obs_buffer, returns_buffer, policy_loss, entropy, value_loss, explained_var = self._policy_phase(
                runner, total_timesteps, ep_info_buf)

            self._auxiliary_phase(obs_buffer, returns_buffer)

            # Periodically log training info
            if update % log_interval == 0 or update == 1 or runner.steps >= total_timesteps:
                logger.log_scalar(runner.steps, 'loss/policy_entropy', entropy)
                logger.log_scalar(runner.steps, 'loss/policy_loss', policy_loss)
                logger.log_scalar(runner.steps, 'loss/value_loss', value_loss)
                logger.log_scalar(runner.steps, 'vf_explained_variance', explained_var)
                logger.log_scalar(runner.steps, 'episode/reward_mean',
                                  safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                logger.log_scalar(runner.steps, 'episode/length_mean',
                                  safe_mean([ep_info['l'] for ep_info in ep_info_buf]))

            # Periodically save model weights
            if (save_interval is not None and update % save_interval == 0) or runner.steps >= total_timesteps:
                self.save(os.path.join(logger.directory, 'weights', f'update_{update}'))

            update += 1

            pbar.update(runner.batch_size * self.policy_iterations)

        pbar.close()

    @tf.function
    def act(self, observation):
        pi = self.policy(observation)
        return pi.mode()

    def load(self, path):
        self.policy.load_weights(path)

    def save(self, path):
        self.policy.save_weights(path)
