from typing import Union, Dict

import gin
import gym
import numpy as np
import tensorflow as tf

from interact.agents.dqn.dueling import DuelingAggregator
from interact.experience.sample_batch import SampleBatch
from interact.networks import build_network_fn
from interact.policies.base import Policy
from interact.utils.math_utils import NormcInitializer

layers = tf.keras.layers


@gin.configurable("qnetwork", allowlist=["dueling", "dueling_hidden_units"])
class QNetwork(tf.keras.Model):
    """A `tf.keras.Model` version of a Q-Network.

    This model has the option of using the dueling network architecture.

    Args:
        observation_space: The observation space of this policy.
        action_space: The action space of this policy.
        network: The type of network to use (e.g. 'cnn', 'mlp').
        dueling: A boolean indicating whether or not to use the dueling architecture.
        dueling_hidden_units: The number of hidden units in the first layer of each
            stream in the dueling network. Only applicable if `dueling` is True.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str,
        dueling: bool = False,
        dueling_hidden_units: int = 64,
    ):
        assert isinstance(
            action_space, gym.spaces.Discrete
        ), "only discrete actions spaces can be used with a QNetwork"

        base_model = build_network_fn(network, observation_space.shape)()
        h = base_model.outputs[0]

        if dueling:
            value_stream = layers.Dense(
                dueling_hidden_units,
                activation="relu",
                kernel_initializer=NormcInitializer(),
            )(h)
            value_stream = layers.Dense(1, kernel_initializer=NormcInitializer(0.01))(
                value_stream
            )

            advantage_stream = layers.Dense(
                dueling_hidden_units,
                activation="relu",
                kernel_initializer=NormcInitializer(),
            )(h)
            advantage_stream = layers.Dense(
                action_space.n, kernel_initializer=NormcInitializer(0.01)
            )(advantage_stream)

            q_values = DuelingAggregator()([value_stream, advantage_stream])
        else:
            q_values = layers.Dense(
                action_space.n, kernel_initializer=NormcInitializer(0.01)
            )(h)

        super().__init__(inputs=base_model.inputs, outputs=[q_values])


class DQNPolicy(Policy):
    """A policy for a DQN agent.

    This policy encapsulates the online Q-network and the target network, and uses
    an epsilon-greedy exploration policy.

    Args:
        observation_space: The observation space of this policy.
        action_space: The action space of this policy.
        network: The type of network to use (e.g. 'cnn', 'mlp').
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str = "cnn",
    ):
        assert isinstance(action_space, gym.spaces.Discrete)
        super().__init__(observation_space, action_space)

        self.q_network = QNetwork(observation_space, action_space, network)
        self.target_network = QNetwork(observation_space, action_space, network)
        self.target_network.trainable = False
        self.update_target_network()

    @tf.function
    def call(self, inputs, **kwargs):
        return self.q_network(inputs)

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    @tf.function
    def _step(
        self, obs: np.ndarray, states: Union[np.ndarray, None] = None, **kwargs
    ) -> Dict[str, Union[float, np.ndarray]]:
        epsilon = kwargs.get("epsilon")

        q_values = self.q_network(obs)
        deterministic_actions = tf.argmax(q_values, axis=-1)
        random_actions = tf.random.uniform(
            [len(obs)], 0, self.action_space.n, dtype=tf.int64
        )
        choose_random = tf.random.uniform([len(obs)], 0, 1) < epsilon
        stochastic_actions = tf.where(
            choose_random, random_actions, deterministic_actions
        )

        return {SampleBatch.ACTIONS: stochastic_actions}
