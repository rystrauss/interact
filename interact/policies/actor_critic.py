from typing import Union, Dict, Callable

import gym
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from interact.experience.sample_batch import SampleBatch
from interact.math_utils import NormcInitializer
from interact.policies.base import Policy

layers = tf.keras.layers


class ActorCriticPolicy(Policy):
    """A generic implementation of an Actor-Critic policy.

    This policy encapsulates both an actor network (for the policy) and critic network (for the value function),
    which optionally share weights.

    Args:
        observation_space: The observation space of this policy.
        action_space: The action space of this policy.
        base_model_fn: A function which returns the model to be used as the base for the policy and value functions.
        value_network: Either 'shared' or 'copy', indicating whether or not the value function should share weights
            with the policy.
    """

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 base_model_fn: Callable[[], layers.Layer],
                 value_network: str = 'copy'):
        super().__init__(observation_space, action_space)

        if value_network == 'copy':
            self._policy_base = base_model_fn()
            self._value_base = base_model_fn()
            self._shared_base = None
        elif value_network == 'shared':
            self._policy_base = None
            self._value_base = None
            self._shared_base = base_model_fn()
        else:
            raise ValueError('`value_network` must be either "copy" or "shared"')

        if isinstance(action_space, gym.spaces.Discrete):
            self._policy_fn = layers.Dense(action_space.n)
            self._is_discrete = True
        else:
            self._policy_fn = layers.Dense(action_space.shape[0], kernel_initializer=NormcInitializer(0.01))
            self._policy_logstds = self.add_weight('policy_logstds', shape=(action_space.shape[0],), trainable=True,
                                                   initializer=tf.keras.initializers.Zeros())
            self._is_discrete = False

        self._value_fn = layers.Dense(1, kernel_initializer=NormcInitializer(0.01))

    def make_pdf(self, latent):
        if self._is_discrete:
            pi = tfd.Categorical(latent)
        else:
            pi = tfd.MultivariateNormalDiag(latent, tf.exp(self._policy_logstds))
        return pi

    def call(self, inputs, **kwargs):
        if self._shared_base is None:
            policy_latent = self._policy_base(inputs)
            value_latent = self._value_base(inputs)
            pi = self.make_pdf(self._policy_fn(policy_latent))
            value_preds = tf.squeeze(self._value_fn(value_latent), axis=-1)
        else:
            shared_latent = self._shared_base(inputs)
            pi = self.make_pdf(self._policy_fn(shared_latent))
            value_preds = tf.squeeze(self._value_fn(shared_latent), axis=-1)

        return pi, value_preds

    @tf.function
    def _step(self,
              obs: np.ndarray,
              states: Union[np.ndarray, None] = None,
              **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        pi, value_preds = self.call(obs)

        actions = pi.sample()
        action_logp = pi.log_prob(actions)

        return {
            SampleBatch.ACTIONS: actions,
            SampleBatch.ACTION_LOGP: action_logp,
            SampleBatch.VALUE_PREDS: value_preds
        }

    @tf.function
    def value(self, inputs, **kwargs):
        return self(inputs, **kwargs)[1]
