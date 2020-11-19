from typing import Union, Dict

import gym
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from interact.experience.sample_batch import SampleBatch
from interact.networks import build_network_fn
from interact.policies.base import Policy
from interact.typing import TensorShape
from interact.utils.math_utils import NormcInitializer

layers = tf.keras.layers


class SACPolicy(Policy):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, network: str):
        super().__init__(observation_space, action_space)

        self._discrete = isinstance(action_space, gym.spaces.Discrete)

        self._base_model = build_network_fn(network, observation_space.shape)()
        num_outputs = action_space.shape[0] * 2 if not self._discrete else action_space.n
        self._policy_fn = layers.Dense(num_outputs, kernel_initializer=NormcInitializer(0.01))

    def call(self, inputs, **kwargs):
        latent = self._base_model(inputs)
        logits = self._policy_fn(latent)

        if self._discrete:
            logits = tf.nn.log_softmax(logits)
            pi = tfd.Categorical(logits)
            actions = pi.sample()
            logpacs = pi.log_prob(actions)
        else:
            means, logstds = tf.split(logits, 2, axis=-1)
            pi = tfd.MultivariateNormalDiag(means, tf.exp(logstds))
            actions = pi.sample()
            logpacs = pi.log_prob(actions)
            #  This formula is mathematically equivalent to
            #  `tf.log1p(-tf.square(tf.tanh(x)))`, however this code is more numerically stable.
            #
            #  Derivation:
            #    log(1 - tanh(x)^2)
            #    = log(sech(x)^2)
            #    = 2 * log(sech(x))
            #    = 2 * log(2e^-x / (e^-2x + 1))
            #    = 2 * (log(2) - x - log(e^-2x + 1))
            #    = 2 * (log(2) - x - softplus(-2x))
            logpacs -= tf.reduce_sum(2 * (np.log(2) - actions - tf.nn.softplus(-2 * actions)), axis=1)
            actions = tf.nn.tanh(actions)

        return actions, logpacs, pi.entropy(), logits

    @tf.function
    def _step(self,
              obs: np.ndarray,
              states: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        actions, logpacs, _, _ = self.call(obs)
        return {
            SampleBatch.ACTIONS: actions,
            SampleBatch.ACTION_LOGP: logpacs
        }


class QFunction(layers.Layer):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 network: str,
                 units: TensorShape = tuple(),
                 **kwargs):
        super().__init__(**kwargs)

        assert isinstance(observation_space, gym.spaces.Box)
        self._discrete = isinstance(action_space, gym.spaces.Discrete)

        self._concat_at_front = not self._discrete and len(observation_space.shape) == 1

        if self._concat_at_front:
            base_model_input_shape = (observation_space.shape[0] + action_space.shape[0],)
        else:
            base_model_input_shape = observation_space.shape

        self._base_model = build_network_fn(network, base_model_input_shape)()

        output_units = 1 if not self._discrete else action_space.n
        self._output = tf.keras.Sequential(
            [layers.Dense(i, kernel_initializer=NormcInitializer()) for i in units] + [
                layers.Dense(output_units, kernel_initializer=NormcInitializer(0.01))])

    def call(self, inputs, **kwargs):
        if self._discrete:
            latent = self._base_model(inputs)
            q = self._output(latent)
        else:
            obs, actions = inputs
            if self._concat_at_front:
                latent = self._base_model(tf.concat([obs, actions], axis=-1))
            else:
                latent = self._base_model(obs)
                latent = tf.concat([latent, actions])

            q = tf.squeeze(self._output(latent), axis=-1)

        return q
