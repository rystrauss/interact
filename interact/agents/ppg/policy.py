from typing import Union, Dict

import gym
import numpy as np
import tensorflow as tf

from interact.experience.sample_batch import SampleBatch
from interact.policies.actor_critic import ActorCriticPolicy
from interact.utils.initialization import NormcInitializer

layers = tf.keras.layers


class PPGPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str,
    ):
        super().__init__(observation_space, action_space, network, value_network="copy")

        self._aux_value_fn = layers.Dense(
            1, kernel_initializer=NormcInitializer(0.01), name="aux_value"
        )

        self.policy_weights = (
            self._policy_base.trainable_weights + self._policy_fn.trainable_weights
        )
        if not self.is_discrete:
            self.policy_weights += self._policy_logstds.trainable_weights
        self.value_weights = (
            self._value_base.trainable_weights + self._value_fn.trainable_weights
        )
        self.policy_and_value_weights = self.policy_weights + self.value_weights
        self.auxiliary_weights = (
            self.policy_weights + self._aux_value_fn.trainable_weights
        )

        self.auxiliary_heads(tf.zeros([1, *observation_space.shape]))

    @tf.function
    def policy_logits(self, obs):
        logits = self._policy_fn(self._policy_base(obs))
        if not self.is_discrete:
            logits = tf.concat([logits, self._policy_logstds], axis=-1)
        return logits

    def auxiliary_heads(self, obs):
        policy_latent = self._policy_base(obs)
        pi = self.make_pdf(self._policy_fn(policy_latent))
        aux_value = tf.squeeze(self._aux_value_fn(policy_latent), axis=-1)
        return pi, aux_value

    def call(self, inputs, **kwargs):
        policy_latent = self._policy_base(inputs)
        pi = self.make_pdf(self._policy_fn(policy_latent))

        return pi

    @tf.function
    def _step(self, obs: np.ndarray, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        pi = self.call(obs)
        value_preds = self.value(obs)

        actions = pi.sample()
        action_logp = pi.log_prob(actions)

        return {
            SampleBatch.ACTIONS: actions,
            SampleBatch.ACTION_LOGP: action_logp,
            SampleBatch.VALUE_PREDS: value_preds,
        }

    @tf.function
    def value(self, inputs, **kwargs):
        value_latent = self._value_base(inputs)
        value_preds = tf.squeeze(self._value_fn(value_latent), axis=-1)
        return value_preds
