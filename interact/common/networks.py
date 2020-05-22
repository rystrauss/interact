"""Module that provides an interface for building networks.

Author: Ryan Strauss
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def build_network(network, input_shape, **network_kwargs):
    """Builds a network of the specified type.

    Args:
        network: The type of network to be built. Currently, only 'mlp' is supported.
        input_shape: The network's input shape. Should correspond to the shape of the environment's observations.
        **network_kwargs: Keyword arguments to be passed to the network building function.

    Returns:
        The specified network, as a `tf.keras.Model`.
    """
    if network == 'mlp':
        return build_mlp(input_shape, **network_kwargs)
    else:
        raise NotImplementedError('mlp is the only supported network type')


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

    layers = [Dense(units[0], activation=activation, input_shape=input_shape)]

    for n in units[1:]:
        layers.append(Dense(n, activation=activation))

    return Sequential(layers)
