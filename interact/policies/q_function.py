from typing import Tuple

import gin
import gym
import tensorflow as tf
from tensorflow.keras import layers

from interact.networks import build_network_fn
from interact.utils.initialization import NormcInitializer


@gin.configurable(allowlist=["dueling", "output_hidden_units"])
class QFunction(layers.Layer):
    """A learnable Q-function.

    This implementation can be used with discrete or continuous action spaces, and
    can use the dueling architecture for discrete action spaces.

    Args:
        observation_space: The observation space of this policy.
        action_space: The action space of this policy.
        network: The type of network to use (e.g. 'cnn', 'mlp').
        dueling: A boolean indicating whether or not to use the dueling architecture.
        output_hidden_units: The number of hidden units in each stream of the dueling
            network if `dueling` is True. Or, the number of hidden units added after
            the base network if actions are concatenated after the base network in the
            case of continuous action spaces (e.g. when the base network is a CNN).
    """

    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            network: str,
            dueling: bool = False,
            output_hidden_units: Tuple[int] = tuple(),
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

        self._dueling = dueling

        if dueling:
            assert (
                self._discrete
            ), "Dueling architecture can only be used with discrete action spaces."

            self._value_stream = tf.keras.Sequential(
                [
                    layers.Dense(i, kernel_initializer=NormcInitializer())
                    for i in output_hidden_units
                ]
                + [layers.Dense(1, kernel_initializer=NormcInitializer(0.01))]
            )
            self._advantage_stream = tf.keras.Sequential(
                [
                    layers.Dense(i, kernel_initializer=NormcInitializer())
                    for i in output_hidden_units
                ]
                + [
                    layers.Dense(
                        action_space.n, kernel_initializer=NormcInitializer(0.01)
                    )
                ]
            )

            self._aggregator = DuelingAggregator()
        else:
            self._output = tf.keras.Sequential(
                [
                    layers.Dense(i, kernel_initializer=NormcInitializer())
                    for i in output_hidden_units
                ]
                + [
                    layers.Dense(
                        output_units, kernel_initializer=NormcInitializer(0.01)
                    )
                ]
            )

        if self._discrete:
            self.call(tf.zeros((1, *observation_space.shape)))
        else:
            self.call(
                [
                    tf.zeros((1, *observation_space.shape)),
                    tf.zeros((1, *action_space.shape)),
                ]
            )

    def call(self, inputs, **kwargs):
        if self._discrete:
            latent = self._base_model(inputs)
            if self._dueling:
                q = self._aggregator(
                    [self._value_stream(latent), self._advantage_stream(latent)]
                )
            else:
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


class TwinQFunction(layers.Layer):
    """Thin wrapper around two "twin" q-functions."""

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        self.q1 = QFunction(*args, **kwargs)
        self.q2 = QFunction(*args, **kwargs)

    def call(self, inputs, **kwargs):
        return self.q1(inputs), self.q2(inputs)


class DuelingAggregator(tf.keras.layers.Layer):
    """Implements the aggregation module of the dueling network architecture.

    This layer accepts two inputs, the value stream and the advantage stream.
    This layer expects the input as a list that looks like
    [value_stream, advantage_stream].
    """

    def __init__(self, **kwargs):
        super(DuelingAggregator, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        value_stream, advantage_stream = inputs
        output_dim = advantage_stream.shape[1]
        value_stream = tf.tile(value_stream, [1, output_dim])
        # This line corresponds to Equation 9 from Wang et. al.
        output = value_stream + (
                advantage_stream - tf.reduce_mean(advantage_stream, axis=-1,
                                                  keepdims=True)
        )
        return output

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(input_shape[1])
