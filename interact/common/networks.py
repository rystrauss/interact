from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def build_network(network, input_shape, **network_kwargs):
    if network == 'mlp':
        return build_mlp(input_shape, **network_kwargs)
    else:
        raise NotImplementedError('mlp is the only supported network type')


def build_mlp(input_shape, units=(64, 64), activation='relu'):
    assert len(units) > 0, 'there must be at least one hidden layer'

    layers = [Dense(units[0], activation=activation, input_shape=input_shape)]

    for n in units[1:]:
        layers.append(Dense(n, activation=activation))

    return Sequential(layers)
