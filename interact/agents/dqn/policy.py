from typing import Union, Dict

import gym
import numpy as np
import tensorflow as tf

from interact.experience.sample_batch import SampleBatch
from interact.policies.base import Policy
from interact.policies.q_function import QFunction

layers = tf.keras.layers


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

        self.q_network = QFunction(observation_space, action_space, network)
        self.target_network = QFunction(observation_space, action_space, network)
        self.target_network.trainable = False

    def build(self, input_shape):
        self.q_network.build(input_shape)
        self.target_network.build(input_shape)
        self.update_target_network()

    @tf.function
    def call(self, inputs, **kwargs):
        return self.q_network(inputs)

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    @tf.function
    def _step(self, obs: np.ndarray, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
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
