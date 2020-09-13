import tensorflow as tf

from interact.common.policies import DisjointActorCriticPolicy

layers = tf.keras.layers


class PPGPolicy(DisjointActorCriticPolicy):

    def __init__(self, action_space, model_fn):
        super().__init__(action_space, model_fn, model_fn)

        self._value_aux = layers.Dense(1)

        self.policy_weights = self._policy_latent.trainable_weights + self._policy_fn.trainable_weights
        self.value_weights = self._value_latent.trainable_weights + self._value_fn.trainable_weights
        self.policy_and_value_weights = self.policy_weights + self.value_weights
        self.auxiliary_weights = self.policy_weights + self._value_aux.trainable_weights

    @tf.function
    def policy_logits(self, obs):
        policy_latent = self._policy_latent(obs)
        return self._policy_fn(policy_latent)

    def auxiliary_heads(self, obs):
        policy_latent = self._policy_latent(obs)
        pi = self.make_pdf(self._policy_fn(policy_latent))
        aux_value = tf.squeeze(self._value_aux(policy_latent), axis=-1)
        return pi, aux_value
