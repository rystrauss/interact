"""Implementation of the advantage actor-critic algorithm.

Author: Ryan Strauss
"""

import os
import sys
from collections import deque
from typing import Tuple

import tensorflow as tf
from tqdm import tqdm

from interact.agents.a2c.runner import Runner
from interact.agents.base import Agent
from interact.agents.util import register
from interact.common.math_util import safe_mean
from interact.common.policies import build_actor_critic_policy
from interact.common.schedules import LinearDecay
from interact.logger import Logger


@register('a2c')
class A2CAgent(Agent):
    """An agent that learns using the advantage actor-critic algorithm.

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
        rho: Discounting factor used by RMSprop.
        epsilon: Constant for numerical stability used by RMSprop.
        **network_kwargs: Keyword arguments to be passed to the policy/value network.
    """

    def __init__(self, *, env, load_path=None, policy_network='mlp', value_network='copy', gamma=0.99, nsteps=5,
                 ent_coef=0.01, vf_coef=0.25, learning_rate=0.0001, lr_schedule='constant', max_grad_norm=0.5, rho=0.99,
                 epsilon=1e-5, **network_kwargs):
        assert lr_schedule in {'linear', 'constant'}, 'lr_schedule must be either "linear" or "constant"'

        self.policy = build_actor_critic_policy(policy_network, value_network, env, **network_kwargs)
        self.gamma = gamma
        self.nsteps = nsteps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.learning_rate = learning_rate
        self.lr_schedule = lr_schedule
        self.max_grad_norm = max_grad_norm
        self.rho = rho
        self.epsilon = epsilon
        self.optimizer = None

        super().__init__(env=env, load_path=load_path)

    @tf.function
    def _train_step(self, obs, returns, actions, values) -> Tuple[float, float, float]:
        """Performs a policy update.

        A standard actor-critic updates is used where the advantage function is used as the baseline.

        Args:
            obs: a collection of observations of environment states
            returns: the returns received from each of the states
            actions: the actions that were selected in each state
            values: the values estimates of each state

        Returns:
            A 3-tuple with the following:
                policy_loss: the loss of the policy network
                value_loss: the loss of the value network
                entropy: the current entropy of the policy
        """
        # Calculate the advantages, which are used as the baseline in the actor-critic update
        advantages = returns - values

        with tf.GradientTape() as tape:
            # Compute the policy for the given observations
            pi = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = tf.reduce_sum(-pi.log_prob(actions), axis=-1)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the individual loss functions
            policy_loss = tf.reduce_mean(advantages * neglogpacs)
            value_loss = tf.reduce_mean((returns - self.policy.value(obs)) ** 2)
            # The final loss to be minimized is a combination of the policy and value losses, in addition
            # to an entropy bonus which can be used to encourage exploration
            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.trainable_weights)
        # Perform gradient clipping
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        return policy_loss, value_loss, entropy

    def learn(self, *, total_timesteps, logger, log_interval=100, save_interval=None):
        assert isinstance(logger, Logger), 'logger must be an instance of the `Logger` class'

        # Create the runner that collects experience
        runner = Runner(self.env, self.policy, self.nsteps, self.gamma)

        # Calculate the number of policy updates that we will perform
        nupdates = total_timesteps // runner.batch_size

        # Create the optimizer for updating network parameters
        if self.lr_schedule == 'linear':
            learning_rate = LinearDecay(self.learning_rate, nupdates)
        else:
            learning_rate = self.learning_rate
        self.optimizer = tf.optimizers.RMSprop(learning_rate=learning_rate, rho=self.rho, epsilon=self.epsilon)

        # Create a buffer that holds information about the 100 most recent episodes
        ep_info_buf = deque([], maxlen=100)

        for update in tqdm(range(1, nupdates + 1), desc='Updates', file=sys.stdout):
            # Collect experience from the environment
            # The rollout is a tuple containing observations, returns, actions, and values
            # This information constitutes a batch of experience that we will learn from
            *rollout, ep_infos = runner.run()
            # Perform a policy update based on the collected experience
            policy_loss, value_loss, entropy = self._train_step(*rollout)

            # Keep track of episode information for logging purposes
            ep_info_buf.extend(ep_infos)

            # Periodically log training info
            if update % log_interval == 0 or update == 1:
                logger.log_scalar(update, 'total_timesteps', runner.steps)
                logger.log_scalar(update, 'loss/policy_entropy', entropy)
                logger.log_scalar(update, 'loss/policy_loss', policy_loss)
                logger.log_scalar(update, 'loss/value_loss', value_loss)

                if len(ep_info_buf) > 0 and len(ep_info_buf[0]) > 0:
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
