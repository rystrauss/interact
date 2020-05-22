"""This module provides classes that define agent policies.

A policy is the function that prescribes which action an agent should take in a given environment state.

Author: Ryan Strauss
"""

from abc import ABC, abstractmethod
from enum import Enum, auto

import tensorflow as tf
from gym.spaces import Discrete
from tensorflow_probability import distributions as tfd

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

    def _makepdf(self, logits):
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
        """Returns the policy for a single step.

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
        latent_model: The network that is shared between the policy and value function.
    """

    def __init__(self, action_space, latent_model):
        super().__init__(action_space)
        assert isinstance(latent_model, tf.keras.Model), 'latent_model must be an instance of tf.keras.Model'

        self._latent = latent_model

        assert len(action_space.shape) <= 1, f'received action space with shape {action_space.shape}'
        num_policy_logits = action_space.n if self.is_discrete else 2 * action_space.shape[0]

        self._policy_fn = layers.Dense(num_policy_logits)
        self._value_fn = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        """Returns the policy's probability distribution."""
        latent = self._latent(inputs)
        pi = self._makepdf(self._policy_fn(latent))
        return pi

    @tf.function
    def value(self, obs):
        """Computes the value estimates for given observations.

        Args:
            obs: The observations to be evaluated.

        Returns:
            The value estimates of the provided observations.
        """
        return self._value_fn(self._latent(obs))

    @tf.function
    def step(self, obs):
        """Uses the policy to compute the requisite information for stepping in the environment.

        Args:
            obs: The environment observations to be evaluated.

        Returns:
            The tuple `(actions, values)` with the corresponding actions and value estimates for the given observations.
        """
        latent = self._latent(obs)
        pi = self._makepdf(self._policy_fn(latent))

        actions = pi.sample()
        values = self._value_fn(latent)
        return actions, values
