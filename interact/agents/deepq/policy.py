"""Defines a deep q-learning policy.

Author: Ryan Strauss
"""

import tensorflow as tf

from interact.common.policies import Policy

layers = tf.keras.layers


def build_q_network(num_actions, latent_model_fn):
    """Builds a Q-network.

    Args:
        num_actions: The number of actions the network should output.
        latent_model_fn: A function that returns the model to be used for performing feature extraction.

    Returns:
        A Q-network as a `tf.keras.Model``.
    """
    model = latent_model_fn()
    q_values = layers.Dense(num_actions)(model.outputs[0])
    return tf.keras.Model(inputs=model.inputs, outputs=[q_values])


class DeepQPolicy(Policy):
    """A policy for the deep q-learning algorithm.

    This policy encapsulates both the online network and the offline target network.
    The target network can be used by directly accessing the `target_network` attribute.

    Args:
        action_space: The action space of this policy.
        latent_model_fn: A function that returns the model to be used for performing feature extraction.
    """

    def __init__(self, action_space, latent_model_fn):
        super().__init__(action_space)
        assert self.is_discrete, 'q-learning only works with discrete action spaces'

        self.q_network = build_q_network(action_space.n, latent_model_fn)
        self.target_network = build_q_network(action_space.n, latent_model_fn)

    def call(self, inputs, training=None, mask=None):
        return self.q_network(inputs)

    @tf.function
    def step(self, obs):
        qvalues = self.call(obs)
        return tf.argmax(qvalues, axis=-1), qvalues, None

    def update_target(self):
        """Updates the parameters of the target network to match the current online Q-network."""
        self.target_network.set_weights(self.q_network.get_weights())
