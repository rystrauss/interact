"""This module provides classes that define agent policies.

A policy is the function that prescribes which action an agent should take in a given environment state.

Author: Ryan Strauss
"""

from abc import ABC, abstractmethod
from enum import Enum, auto

import tensorflow as tf
from gym.spaces import Discrete
from tensorflow_probability import distributions as tfd

from interact.common.networks import build_network_fn

layers = tf.keras.layers


class PolicyType(Enum):
    """A policy's type defines whether it maps to a discrete action space or a continuous one."""
    CATEGORICAL = auto()
    CONTINUOUS = auto()


class Policy(tf.keras.Model, ABC):
    """The abstract base class the all policies inherit from.

    Args:
        action_space: The action space of this policy.
    """

    def __init__(self, action_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = action_space
        self._type = PolicyType.CATEGORICAL if isinstance(action_space, Discrete) else PolicyType.CONTINUOUS

    @property
    def type(self) -> PolicyType:
        """The type of this policy."""
        return self._type

    @property
    def is_discrete(self):
        """A boolean indication of whether or not this policy is discrete."""
        return self._type == PolicyType.CATEGORICAL

    def make_pdf(self, logits):
        """Constructs a probability distribution for this policy from a set of logits.

        Args:
            logits: A tensor of logits from which to construct the policy distribution.

        Returns:
            A probability distribution that represents the policy.
        """
        if self.is_discrete:
            pi = tfd.Categorical(logits)
        else:
            mean, logstd = tf.split(logits, 2, axis=-1)
            pi = tfd.Normal(mean, tf.exp(logstd))
        return pi

    @abstractmethod
    def step(self, obs):
        """Computes data relevant for a single step in the environment.

        Args:
            obs: the current observations of the environment

        Returns:
            (actions, values, neglogpacs)
        """
        pass


class ActorCriticPolicy(Policy, ABC):
    """The abstract base class from which actor-critic policies inherit."""

    @abstractmethod
    def value(self, obs):
        """Returns the value estimates of the provided observations.

        Args:
            obs: The observations to be evaluated.

        Returns:
            The value estimates of the provided observations.
        """
        pass


class SharedActorCriticPolicy(ActorCriticPolicy):
    """An actor-critic policy where the value function and policy share hidden layers.

    Thus, this policy only requires that a single model be provided. The policy distribution and the value estimate
    are then both constructed from the features provided by this model.

    Args:
        action_space: The action space of this policy.
        latent_model_fn: A function which returns the network that is shared between the policy and value function.
    """

    def __init__(self, action_space, latent_model_fn):
        super().__init__(action_space)

        self._latent = latent_model_fn()
        assert isinstance(self._latent, tf.keras.Model), 'latent_model_fn must return an instance of tf.keras.Model'

        assert len(action_space.shape) <= 1, f'received action space with shape {action_space.shape}'
        num_policy_logits = action_space.n if self.is_discrete else 2 * action_space.shape[0]

        self._policy_fn = layers.Dense(num_policy_logits)
        self._value_fn = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """Returns the policy's probability distribution."""
        latent = self._latent(inputs)
        pi = self.make_pdf(self._policy_fn(latent))
        return pi

    @tf.function
    def value(self, obs):
        """Computes the value estimates for given observations.

        Args:
            obs: The observations to be evaluated.

        Returns:
            The value estimates of the provided observations.
        """
        return tf.squeeze(self._value_fn(self._latent(obs)), axis=-1)

    @tf.function
    def step(self, obs):
        """Uses the policy to compute the requisite information for stepping in the environment.

        Args:
            obs: The environment observations to be evaluated.

        Returns:
            The tuple `(actions, values, neglogpacs)` with the corresponding actions, value estimates, and negative
            action log probabilities for the given observations.
        """
        latent = self._latent(obs)
        pi = self.make_pdf(self._policy_fn(latent))

        actions = pi.sample()
        values = tf.squeeze(self._value_fn(latent), axis=-1)
        neglogpacs = tf.reduce_sum(tf.reshape(-pi.log_prob(actions), (len(actions), -1)), axis=-1)

        return actions, values, neglogpacs


class DisjointActorCriticPolicy(ActorCriticPolicy):
    """An actor-critic policy where the hidden layers of the value function and policy are distinct.

    Thus, this policy requires that two networks be provided.

    Args:
        action_space: The action space of this policy.
        policy_model_fn: A function which returns the network that is used for the policy.
        value_model_fn: A function which returns the network that is used for the value function.
    """

    def __init__(self, action_space, policy_model_fn, value_model_fn):
        super().__init__(action_space)

        self._policy_latent = policy_model_fn()
        assert isinstance(self._policy_latent,
                          tf.keras.Model), 'policy_model_fn must return an instance of tf.keras.Model'

        self._value_latent = value_model_fn()
        assert isinstance(self._value_latent,
                          tf.keras.Model), 'value_model_fn must return an instance of tf.keras.Model'

        assert len(action_space.shape) <= 1, f'received action space with shape {action_space.shape}'
        num_policy_logits = action_space.n if self.is_discrete else 2 * action_space.shape[0]

        self._policy_fn = layers.Dense(num_policy_logits)
        self._value_fn = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """Returns the policy's probability distribution."""
        policy_latent = self._policy_latent(inputs)
        pi = self.make_pdf(self._policy_fn(policy_latent))
        return pi

    @tf.function
    def value(self, obs):
        """Computes the value estimates for given observations.

        Args:
            obs: The observations to be evaluated.

        Returns:
            The value estimates of the provided observations.
        """
        return tf.squeeze(self._value_fn(self._value_latent(obs)), axis=-1)

    @tf.function
    def step(self, obs):
        """Uses the policy to compute the requisite information for stepping in the environment.

        Args:
            obs: The environment observations to be evaluated.

        Returns:
            The tuple `(actions, values, neglogpacs)` with the corresponding actions, value estimates, and negative
            action log probabilities for the given observations.
        """
        policy_latent = self._policy_latent(obs)
        pi = self.make_pdf(self._policy_fn(policy_latent))
        actions = pi.sample()

        value_latent = self._value_latent(obs)
        values = tf.squeeze(self._value_fn(value_latent), axis=-1)
        neglogpacs = tf.reduce_sum(tf.reshape(-pi.log_prob(actions), (len(actions), -1)), axis=-1)

        return actions, values, neglogpacs


class CopyActorCriticPolicy(DisjointActorCriticPolicy):
    """An actor-critic policy where the hidden layers of the value function and policy are copies of the same network.

    Thus, no parameters are shared between the policy and value function.

    Args:
        action_space: The action space of this policy.
        latent_model_fn: A function which returns the network that is used for the policy and for the value function.
    """

    def __init__(self, action_space, latent_model_fn):
        super().__init__(action_space, latent_model_fn, latent_model_fn)


def build_actor_critic_policy(policy_network, value_network, env, **network_kwargs):
    """Builds a policy for an actor-critic agent.

    Args:
        policy_network: The type of policy network to use.
        value_network: The method of constructing the value network.
        env: The environment that this policy is for.
        **network_kwargs: Keyword arguments to be passed to the network builder.

    Returns:
        The specified actor-critic policy.
    """
    if value_network == 'shared':
        return SharedActorCriticPolicy(
            env.action_space,
            build_network_fn(policy_network, env.observation_space.shape, **network_kwargs))

    if value_network == 'copy':
        return CopyActorCriticPolicy(
            env.action_space,
            build_network_fn(policy_network, env.observation_space.shape, **network_kwargs))

    raise NotImplementedError('value_network must be "shared" or "copy"')
