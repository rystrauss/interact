from typing import Union, Dict, Iterable

import gym
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from interact.experience.sample_batch import SampleBatch
from interact.networks import build_network_fn
from interact.policies.base import Policy
from interact.utils import MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT, SMALL_NUMBER
from interact.utils.math_utils import NormcInitializer

layers = tf.keras.layers


class SACPolicy(Policy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str,
    ):
        super().__init__(observation_space, action_space)

        self._discrete = isinstance(action_space, gym.spaces.Discrete)
        # This assumes that all action dimensions have the same bounds.
        self._action_limit = None if self._discrete else action_space.high[0]

        self._base_model = build_network_fn(network, observation_space.shape)()
        num_outputs = (
            action_space.shape[0] * 2 if not self._discrete else action_space.n
        )
        self._policy_fn = layers.Dense(
            num_outputs, kernel_initializer=NormcInitializer(0.01)
        )

    def call(self, inputs, **kwargs):
        latent = self._base_model(inputs)
        logits = self._policy_fn(latent)

        deterministic = kwargs.get("deterministic", False)

        if self._discrete:
            pi = tfd.Categorical(logits)
            if deterministic:
                actions = pi.mode()
            else:
                actions = pi.sample()
            logpacs = tf.nn.log_softmax(logits)
        else:
            means, logstds = tf.split(logits, 2, axis=-1)
            logstds = tf.clip_by_value(logstds, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)

            pi = tfd.MultivariateNormalDiag(means, tf.exp(logstds))
            if deterministic:
                actions = pi.mean()
            else:
                actions = pi.sample()
            logpacs = pi.log_prob(actions)

            # Adjust loglikelihoods for squashed actions
            logpacs -= tf.reduce_sum(
                2 * (np.log(2) - actions - tf.nn.softplus(-2 * actions)), axis=1
            )

            actions = tf.math.tanh(actions)
            actions = tf.clip_by_value(actions, -1 + SMALL_NUMBER, 1 - SMALL_NUMBER)
            actions *= self._action_limit

        return actions, logpacs

    @tf.function
    def _step(
        self, obs: np.ndarray, states: Union[np.ndarray, None] = None, **kwargs
    ) -> Dict[str, Union[float, np.ndarray]]:
        if kwargs.get("uniform_sample", False):
            if self._discrete:
                actions = tf.random.uniform(
                    [len(obs)], 0, self.action_space.n, dtype=tf.int32
                )
            else:
                actions = tf.random.uniform(
                    [len(obs)], self.action_space.low, self.action_space.high
                )
                actions = tf.reshape(actions, [len(obs), -1])
        else:
            actions, _ = self.call(obs)
        return {SampleBatch.ACTIONS: actions}


class SACQFunction(layers.Layer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str,
        units: Iterable[int] = tuple(),
        **kwargs
    ):
        super().__init__(**kwargs)

        assert isinstance(observation_space, gym.spaces.Box)
        self._discrete = isinstance(action_space, gym.spaces.Discrete)

        self._concat_at_front = not self._discrete and len(observation_space.shape) == 1

        if self._concat_at_front:
            base_model_input_shape = (
                observation_space.shape[0] + action_space.shape[0],
            )
        else:
            base_model_input_shape = observation_space.shape

        self._base_model = build_network_fn(network, base_model_input_shape)()

        output_units = 1 if not self._discrete else action_space.n
        self._output = tf.keras.Sequential(
            [layers.Dense(i, kernel_initializer=NormcInitializer()) for i in units]
            + [layers.Dense(output_units, kernel_initializer=NormcInitializer(0.01))]
        )

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


class TwinQNetwork(layers.Layer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str,
        units: Iterable[int] = tuple(),
        **kwargs
    ):
        super().__init__(**kwargs)

        self.q1 = SACQFunction(observation_space, action_space, network, units)
        self.q2 = SACQFunction(observation_space, action_space, network, units)

    def call(self, inputs, **kwargs):
        return self.q1(inputs), self.q2(inputs)
