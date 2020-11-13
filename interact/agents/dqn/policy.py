from typing import Union, Dict

import gin
import gym
import numpy as np
import tensorflow as tf

from interact.agents.dqn.dueling import DuelingAggregator
from interact.experience.sample_batch import SampleBatch
from interact.networks import build_network_fn
from interact.policies.base import Policy

layers = tf.keras.layers


@gin.configurable('qnetwork', whitelist=['dueling', 'dueling_units'])
class QNetwork(tf.keras.Model):

    def __init__(self,
                 observation_space: gym.Space,
                 action_space: gym.Space,
                 network: str,
                 dueling: bool = False,
                 dueling_units: int = 64):
        assert isinstance(action_space, gym.spaces.Discrete), 'only discrete actions spaces can be used with a QNetwork'

        base_model = build_network_fn(network, observation_space.shape)()
        h = base_model.outputs[0]

        if dueling:
            value_stream = layers.Dense(dueling_units, activation='relu')(h)
            value_stream = layers.Dense(1)(value_stream)

            advantage_stream = layers.Dense(dueling_units, activation='relu')(h)
            advantage_stream = layers.Dense(action_space.n)(advantage_stream)

            q_values = DuelingAggregator()([value_stream, advantage_stream])
        else:
            q_values = layers.Dense(action_space.n)(h)

        super().__init__(inputs=base_model.inputs, outputs=[q_values])


class DQNPolicy(Policy):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, network: str = 'cnn'):
        super().__init__(observation_space, action_space)

        self.q_network = QNetwork(observation_space, action_space, network)
        self.target_network = QNetwork(observation_space, action_space, network)

    @tf.function
    def call(self, inputs, **kwargs):
        return self.q_network(inputs)

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def _step(self,
              obs: np.ndarray,
              states: Union[np.ndarray, None] = None,
              **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        epsilon = kwargs.get('epsilon')

        q_values = self(obs)
        actions = tf.argmax(q_values, axis=-1).numpy()

        for i in range(len(actions)):
            if np.random.rand() < epsilon:
                actions[i] = self.action_space.sample()

        return {
            SampleBatch.ACTIONS: actions
        }
