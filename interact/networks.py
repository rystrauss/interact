from typing import Callable, Union, List, Tuple

import gin
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Lambda, Permute

from interact.typing import TensorShape
from interact.utils.math_utils import NormcInitializer

_mapping = {}


def register(name):
    """Registers a network type so it can be accessed through the CLI."""

    def _thunk(func):
        _mapping[name] = func
        return func

    return _thunk


def build_network_fn(
    network: str, input_shape: TensorShape
) -> Callable[[], tf.keras.Model]:
    """Returns a function that builds a network of the specified type.

    Args:
        network: The type of network to be built.
        input_shape: The network's input shape. Should correspond to the shape of the
            environment's observations.

    Returns:
        A function that returns the specified network, as a `tf.keras.Model`.
    """
    if network not in _mapping:
        raise NotImplementedError(f"{network} is not a supported network type")

    builder_fn = _mapping[network]

    return lambda: builder_fn(input_shape)


@gin.configurable(name_or_fn="mlp", blacklist=["input_shape"])
@register("mlp")
def build_mlp(
    input_shape: TensorShape,
    hidden_units: Union[List[int], Tuple[int]] = (64, 64),
    activation: str = "relu",
) -> tf.keras.Model:
    """Build a fully-connected feed-forward network, or multilayer-perceptron.

    Args:
        input_shape: The network's input shape.
        hidden_units: An iterable of integers where the ith number is the number of
            units in the ith hidden layer.
        activation: The activation function to be used in the network.

    Returns:
        The specified MLP, as a `tf.keras.Model`.
    """
    assert len(hidden_units) > 0, "there must be at least one hidden layer"

    layers = [Flatten(input_shape=input_shape)]

    for n in hidden_units:
        layers.append(
            Dense(n, activation=activation, kernel_initializer=NormcInitializer())
        )

    return Sequential(layers)


@gin.configurable(name_or_fn="cnn", blacklist=["input_shape"])
@register("cnn")
def build_nature_cnn(
    input_shape: TensorShape,
    scale_inputs: bool = True,
    hidden_units: Union[List[int], Tuple[int]] = (512,),
    permute_channels: bool = True,
) -> tf.keras.Model:
    """Builds a convolutional neural network.

    Defaults to the network described in the DQN Nature article.

    Args:
        input_shape: The network's input shape.
        scale_inputs: If True, model inputs (which are in this case assumed to be 8-bit
            ints) will be scaled to the range [0,1] in the first layer.
        hidden_units: An iterable of integers where the ith number is the number of
            units in the ith dense layer after the convolutional layers.
        permute_channels: If True, inputs are expected to have the format BCHW and will
            be permuted to BHWC format.

    Returns:
        The specified CNN, as a `tf.keras.Model`.
    """
    #
    front_layers = []

    if permute_channels:
        front_layers.append(Permute((2, 3, 1)))

    if scale_inputs:
        front_layers.append(Lambda(lambda x: tf.cast(x, tf.float32) / 255))

    dense_layers = [
        Dense(n, activation="relu", kernel_initializer=NormcInitializer())
        for n in hidden_units
    ]

    model = Sequential(
        [
            *front_layers,
            Conv2D(32, 8, 4, activation="relu"),
            Conv2D(64, 4, 2, activation="relu"),
            Conv2D(64, 3, 1, activation="relu"),
            Flatten(),
            *dense_layers,
        ]
    )

    model.build((None, *input_shape))

    return model
