"""Module that provides an interface for building networks.

Author: Ryan Strauss
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Orthogonal
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Lambda

_mapping = {}


def register(name):
    """Decorator that registers a network type so it can be accessed through the command line interface."""

    def _thunk(func):
        _mapping[name] = func
        return func

    return _thunk


def build_network_fn(network, input_shape, **network_kwargs):
    """Returns a function that builds a network of the specified type.

    Args:
        network: The type of network to be built.
        input_shape: The network's input shape. Should correspond to the shape of the environment's observations.
        **network_kwargs: Keyword arguments to be passed to the network building function.

    Returns:
        A function that returns the specified network, as a `tf.keras.Model`.
    """
    if network not in _mapping:
        raise NotImplementedError(f'{network} is not a supported network type')

    builder_fn = _mapping[network]

    return lambda: builder_fn(input_shape, **network_kwargs)


@register('mlp')
def build_mlp(input_shape, units=(64, 64), activation='relu'):
    """Build a fully-connected feed-forward network, or multilayer-perceptron.

    Args:
        input_shape: The network's input shape.
        units: An iterable of integers where the ith number is the number of units in the ith hidden layer.
        activation: The activation function to be used in the network.

    Returns:
        The specified MLP, as a `tf.keras.Model`.
    """
    assert len(units) > 0, 'there must be at least one hidden layer'

    layers = [Flatten(input_shape=input_shape)]

    for n in units:
        layers.append(Dense(n, activation=activation, kernel_initializer=Orthogonal(np.sqrt(2))))

    return Sequential(layers)


@register('cnn')
def build_nature_cnn(input_shape, scale_inputs=True, units=(512,)):
    """Builds a convolutional neural network.

    Defaults to the network described in the DQN Nature article.

    Args:
        input_shape: The network's input shape.
        scale_inputs: If True, model inputs (which are in this case assumed to be 8-bit ints) will be scaled
            to the range [0,1] in the first layer.
        units: An iterable of integers where the ith number is the number of units in the ith dense layer after
            the convolutional layers.

    Returns:
        The specified CNN, as a `tf.keras.Model`.
    """
    if scale_inputs:
        front_layers = [
            Lambda(lambda x: tf.cast(x, tf.float32) / 255, input_shape=input_shape),
            Conv2D(32, 8, 4, activation='relu')
        ]
    else:
        front_layers = [Conv2D(32, 8, 4, activation='relu', input_shape=input_shape)]

    dense_layers = [Dense(n, activation='relu') for n in units]

    return Sequential([
        *front_layers,
        Conv2D(64, 4, 2, activation='relu'),
        Conv2D(64, 3, 1, activation='relu'),
        Flatten(),
        *dense_layers
    ])
