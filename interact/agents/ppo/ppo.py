"""Implementation of the Proximal Policy Optimization algorithm.

Author: Ryan Strauss
"""

import os
import sys
from collections import deque

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from interact.agents.base import Agent
from interact.agents.ppo.runner import Runner
from interact.agents.util import register
from interact.common.math_util import safe_mean, explained_variance
from interact.common.policies import build_actor_critic_policy
from interact.common.schedules import LinearDecay
from interact.logger import Logger


@register('ppo')
class PPOAgent(Agent):
    """An agent that learns using the Proximal Policy Optimization algorithm.

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

    def __init__(self, *, env, load_path=None, policy_network='mlp', value_network='copy', gamma=0.99, nsteps=2048,
                 ent_coef=0.0, vf_coef=0.5, learning_rate=3e-4, lr_schedule='constant', max_grad_norm=0.5, lam=0.95,
                 nminibatches=4, noptepochs=4, cliprange=0.2, cliprange_schedule='constant', **network_kwargs):
        assert lr_schedule in {'linear', 'constant'}, 'lr_schedule must be either "linear" or "constant"'

        self.policy = build_actor_critic_policy(policy_network, value_network, env, **network_kwargs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.max_grad_norm = max_grad_norm
        self.lam = lam
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.cliprange = cliprange
        self.cliprange_schedule = cliprange_schedule
        self.optimizer = None

        super().__init__(env=env, load_path=load_path)

    @tf.function
    def _train_step(self, obs, returns, actions, values, neglogpacs_old, cliprange):
        """Performs a policy update.

        A standard actor-critic updates is used where the advantage function is used as the baseline.

        Args:
            obs: a collection of observations of environment states
            returns: the returns received from each of the states
            actions: the actions that were selected in each state
            values: the values estimates of each state
            neglogpacs_old: the negative log probabilities of the actions from the old policy
            cliprange: The current value of the clipping term in the surrogate loss.

        Returns:
            A 4-tuple with the following:
                policy_loss: the loss of the policy network
                value_loss: the loss of the value network
                entropy: the current entropy of the policy
                clipfrac: the fraction of the actions which were clipped
        """
        # Calculate the advantages, which are used as the baseline in the actor-critic update
        advantages = returns - values
        # Normalize the advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        with tf.GradientTape() as tape:
            # Compute the policy for the given observations
            pi = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = -pi.log_prob(actions)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the policy surrogate loss
            ratio = tf.exp(neglogpacs_old - neglogpacs)
            pg_loss_unclipped = -advantages * ratio
            pg_loss_clipped = -advantages * tf.clip_by_value(ratio, 1 - cliprange, 1 + cliprange)
            policy_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
            # Define the value loss
            value_preds = self.policy.value(obs)
            value_preds_clipped = tf.clip_by_value(value_preds, -cliprange, cliprange)
            vf_loss_unclipped = (returns - value_preds) ** 2
            vf_loss_clipped = (returns - value_preds_clipped) ** 2
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(vf_loss_clipped, vf_loss_unclipped))
            # The final loss to be minimized is a combination of the policy and value losses, in addition
            # to an entropy bonus which can be used to encourage exploration
            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

        clipfrac = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32))

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.trainable_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        return policy_loss, value_loss, entropy, clipfrac

    def learn(self, *, total_timesteps, logger, log_interval=100, save_interval=None):
        assert isinstance(logger, Logger), 'logger must be an instance of the `Logger` class'

        # Create the runner that collects experience
        runner = Runner(self.env, self.policy, self.nsteps, self.gamma, self.lam)

        # Calculate the number of policy updates that we will perform
        nupdates = total_timesteps // runner.batch_size

        # Create the optimizer for updating network parameters
        if self.lr_schedule == 'linear':
            learning_rate = LinearDecay(self.learning_rate, nupdates)
        elif self.lr_schedule == 'constant':
            learning_rate = self.learning_rate
        else:
            raise ValueError('lr_schedule must be either "linear" or "constant"')
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        if self.cliprange_schedule == 'linear':
            cliprange = LinearDecay(self.cliprange, nupdates)
        elif self.cliprange_schedule == 'constant':
            cliprange = self.cliprange
        else:
            raise ValueError('cliprange_schedule must be either "linear" or "constant"')

        # Create a buffer that holds information about the 100 most recent episodes
        ep_info_buf = deque([], maxlen=100)

        nbatch = runner.batch_size
        nbatch_train = nbatch // self.nminibatches

        for update in tqdm(range(1, nupdates + 1), desc='Updates', file=sys.stdout):
            # Collect experience from the environment
            # The rollout is a tuple containing observations, returns, actions, and values
            # This information constitutes a batch of experience that we will learn from
            *rollout, ep_infos = runner.run()

            # Compute the current clipping amount
            cur_cliprange = cliprange if self.cliprange_schedule == 'constant' else cliprange(update)

            # Perform a policy update based on the collected experience
            # The batch of experience is split into smaller minibatches, and we perform several
            # epochs over those minibatches
            mb_losses = []
            indices = np.arange(runner.batch_size)
            for _ in range(self.noptepochs):
                np.random.shuffle(indices)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mb_indices = indices[start:end]
                    rollout_slices = (x[mb_indices] for x in (*rollout,))
                    mb_losses.append(self._train_step(*rollout_slices, cur_cliprange))

            # Keep track of episode information for logging purposes
            ep_info_buf.extend(ep_infos)

            policy_loss, value_loss, entropy, clipfrac = np.mean(mb_losses, axis=0).tolist()

            # Periodically log training info
            if update % log_interval == 0 or update == 1:
                logger.log_scalar(update, 'total_timesteps', runner.steps)
                logger.log_scalar(update, 'loss/policy_entropy', entropy)
                logger.log_scalar(update, 'loss/policy_loss', policy_loss)
                logger.log_scalar(update, 'loss/value_loss', value_loss)
                logger.log_scalar(update, 'clipfrac', clipfrac)
                logger.log_scalar(update, 'vf_explained_variance', explained_variance(rollout[3], rollout[1]))
                logger.log_scalar(update, 'episode/reward_mean',
                                  safe_mean([ep_info['r'] for ep_info in ep_info_buf]))
                logger.log_scalar(update, 'episode/length_mean',
                                  safe_mean([ep_info['l'] for ep_info in ep_info_buf]))

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
