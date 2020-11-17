from typing import Union, Dict, Callable

import gym
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from interact.experience.sample_batch import SampleBatch
from interact.utils.math_utils import NormcInitializer
from interact.networks import build_network_fn
from interact.policies.base import Policy
from interact.typing import TensorShape, TensorType

layers = tf.keras.layers


class SquashedGaussianActor(layers.Layer):

    def __init__(self,
                 action_space: gym.Space,
                 base_model_fn: Callable[[], layers.Layer],
                 action_limit: Union[TensorType, float] = 1.0,
                 **kwargs):
        super().__init__(**kwargs)

        self._base_model = base_model_fn()
        self._policy_fn = layers.Dense(action_space.shape[0] * 2, kernel_initializer=NormcInitializer(0.01))
        self._action_limit = action_limit

    def call(self, inputs, **kwargs):
        latent = self._base_model(inputs)
        means, logstds = tf.split(self._policy_fn(latent), 2, axis=-1)

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
        actions *= self._action_limit[None, ...]

        return actions, logpacs, pi.entropy()


class QFunction(layers.Layer):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 network: str,
                 units: TensorShape = tuple(),
                 **kwargs):
        super().__init__(**kwargs)

        assert isinstance(observation_space, gym.spaces.Box)

        self._concat_at_front = len(observation_space.shape) == 1

        if self._concat_at_front:
            base_model_input_shape = (observation_space.shape[0] + action_space.shape[0],)
        else:
            base_model_input_shape = observation_space.shape

        self._base_model = build_network_fn(network, base_model_input_shape)()
        self._output = tf.keras.Sequential(
            [layers.Dense(i, kernel_initializer=NormcInitializer(0.01)) for i in units] + [
                layers.Dense(1, kernel_initializer=NormcInitializer(0.01))])

    def call(self, inputs, **kwargs):
        obs, actions = inputs

        if self._concat_at_front:
            latent = self._base_model(tf.concat([obs, actions], axis=-1))
        else:
            latent = self._base_model(obs)
            latent = tf.concat([latent, actions])

        q = self._output(latent)
        return tf.squeeze(q, axis=-1)


class SACPolicy(Policy):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 network: str,
                 actor_only: bool = False):
        super().__init__(observation_space, action_space)

        self.actor_only = actor_only

        base_model_fn = build_network_fn(network, observation_space.shape)

        self.pi = SquashedGaussianActor(action_space, base_model_fn, action_space.high, name='actor')

        if not self.actor_only:
            self.q1 = QFunction(observation_space, action_space, network, name='q1')
            self.q2 = QFunction(observation_space, action_space, network, name='q2')
        else:
            self.q1 = None
            self.q2 = None

    @property
    def q_variables(self):
        if self.actor_only:
            return []

        return self.q1.variables + self.q1.variables

    def act(self, obs: np.ndarray):
        return self.pi(obs)[0]

    @tf.function
    def _step(self,
              obs: np.ndarray,
              states: Union[np.ndarray, None] = None, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        actions, logpacs, _ = self.pi(obs)
        return {
            SampleBatch.ACTIONS: actions,
            SampleBatch.ACTION_LOGP: logpacs
        }
