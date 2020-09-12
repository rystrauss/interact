import tensorflow as tf

from interact.common.policies import DisjointActorCriticPolicy

layers = tf.keras.layers


class PPGPolicy(DisjointActorCriticPolicy):

    def __init__(self, action_space, model_fn):
        super().__init__(action_space, model_fn, model_fn)

        self._value_aux = layers.Dense(1)

        self.policy_trainable_weights = self._policy_latent.trainable_weights + self._policy_fn.trainable_weights
