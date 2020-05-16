from abc import ABC, abstractmethod
from enum import Enum, auto

import tensorflow as tf
from gym.spaces import Discrete
from tensorflow_probability import distributions as tfd

layers = tf.keras.layers


class PolicyType(Enum):
    CATEGORICAL = auto()
    CONTINUOUS = auto()


class Policy(tf.keras.Model, ABC):

    def __init__(self, action_space, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = action_space
        self._type = PolicyType.CATEGORICAL if isinstance(action_space, Discrete) else PolicyType.CONTINUOUS

    @property
    def type(self) -> PolicyType:
        return self._type

    @property
    def is_discrete(self):
        return self._type == PolicyType.CATEGORICAL

    def _makepdf(self, logits):
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

    @abstractmethod
    def value(self, obs):
        pass


class SharedActorCriticPolicy(ActorCriticPolicy):

    def __init__(self, action_space, latent_model):
        super().__init__(action_space)
        assert isinstance(latent_model, tf.keras.Model), 'latent_model must be an instance of tf.keras.Model'

        self._latent = latent_model

        assert len(action_space.shape) <= 1, f'received action space with shape {action_space.shape}'
        num_policy_logits = action_space.n if self.is_discrete else 2 * action_space.shape[0]

        self._policy_fn = layers.Dense(num_policy_logits)
        self._value_fn = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        latent = self._latent(inputs)
        pi = self._makepdf(self._policy_fn(latent))

        actions = pi.sample()
        values = self._value_fn(latent)
        neglogpacs = -pi.log_prob(actions)
        return actions, values, neglogpacs

    def value(self, obs):
        return self._value_fn(self._latent(obs))

    def step(self, obs):
        return self.call(obs)
