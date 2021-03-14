from typing import Union, Dict

import gin
import gym
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from interact.experience.sample_batch import SampleBatch
from interact.networks import build_network_fn
from interact.policies.base import Policy
from interact.policies.q_function import QFunction, TwinQFunction
from interact.utils.initialization import NormcInitializer

layers = tf.keras.layers


@gin.configurable(allowlist=["free_log_std"])
class ActorCriticPolicy(Policy):
    """A generic implementation of an Actor-Critic policy.

    This policy encapsulates both an actor network (for the policy) and critic network
    (for the value function), which optionally share weights.

    Args:
        observation_space: The observation space of this policy.
        action_space: The action space of this policy.
        network: The type of network to be built.
        value_network: Either 'shared' or 'copy', indicating whether or not the value
            function should share weights with the policy.
        free_log_std: Use free-floating (i.e. non-state-dependent) variables for the
            policy scales in continuous actions spaces.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str,
        value_network: str = "copy",
        free_log_std: bool = True,
    ):
        super().__init__(observation_space, action_space)

        network_fn = build_network_fn(network, observation_space.shape)

        if value_network == "copy":
            self._policy_base = network_fn()
            self._value_base = network_fn()
            self._shared_base = None
        elif value_network == "shared":
            self._policy_base = None
            self._value_base = None
            self._shared_base = network_fn()
        else:
            raise ValueError('`value_network` must be either "copy" or "shared"')

        if isinstance(action_space, gym.spaces.Discrete):
            self._policy_fn = layers.Dense(action_space.n)
            self.is_discrete = True
        else:
            if free_log_std:
                self._policy_logstds = self.add_weight(
                    "policy_logstds",
                    shape=(action_space.shape[0],),
                    trainable=True,
                    initializer=tf.keras.initializers.Zeros(),
                )
                self._policy_fn = layers.Dense(
                    action_space.shape[0], kernel_initializer=NormcInitializer(0.01)
                )
            else:
                self._policy_logstds = None
                self._policy_fn = layers.Dense(
                    action_space.shape[0] * 2, kernel_initializer=NormcInitializer(0.01)
                )
            self.is_discrete = False

        self._value_fn = layers.Dense(1, kernel_initializer=NormcInitializer(0.01))

        self.call(tf.zeros((1, *observation_space.shape)))

    def make_pdf(self, latent):
        if self.is_discrete:
            pi = tfd.Categorical(latent)
        else:
            if self._policy_logstds is not None:
                mu = latent
                logstd = self._policy_logstds
            else:
                mu, logstd = tf.split(latent, 2, axis=-1)

            pi = tfd.MultivariateNormalDiag(mu, tf.exp(logstd))
        return pi

    def call(self, inputs):
        if self._shared_base is None:
            policy_latent = self._policy_base(inputs)
            value_latent = self._value_base(inputs)
            pi = self.make_pdf(self._policy_fn(policy_latent))
            value_preds = tf.squeeze(self._value_fn(value_latent), axis=-1)
        else:
            shared_latent = self._shared_base(inputs)
            pi = self.make_pdf(self._policy_fn(shared_latent))
            value_preds = tf.squeeze(self._value_fn(shared_latent), axis=-1)

        return pi, value_preds

    @tf.function
    def _step(self, obs: np.ndarray, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        pi, value_preds = self.call(obs)

        actions = pi.sample()
        action_logp = pi.log_prob(actions)

        return {
            SampleBatch.ACTIONS: actions,
            SampleBatch.ACTION_LOGP: action_logp,
            SampleBatch.VALUE_PREDS: value_preds,
        }

    @tf.function
    def value(self, inputs):
        return self(inputs)[1]


class DeterministicActorCriticPolicy(Policy):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        network: str,
        use_twin_critic: bool = False,
    ):
        super().__init__(observation_space, action_space)
        assert isinstance(action_space, gym.spaces.Box), (
            "DeterministicActorCriticPolicy can only be used with "
            "continuous action spaces."
        )

        self._action_space_low = action_space.low[np.newaxis]
        self._action_space_high = action_space.high[np.newaxis]

        network_fn = build_network_fn(network, observation_space.shape)

        def squash(x):
            x = tf.nn.sigmoid(2 * x)
            return (
                self._action_space_high - self._action_space_low
            ) * x + self._action_space_low

        policy_layers = [
            network_fn(),
            layers.Dense(action_space.shape[0]),
        ]

        bounded = np.logical_and(
            action_space.bounded_above, action_space.bounded_below
        ).any()

        if bounded:
            policy_layers.append(layers.Lambda(squash))

        self.policy = tf.keras.Sequential(policy_layers)

        q_class = TwinQFunction if use_twin_critic else QFunction
        self.q_function = q_class(observation_space, action_space, network)

        self.policy.build([None, *observation_space.shape])

    def call(self, inputs, **kwargs):
        noise_scale = kwargs.get("noise_scale", 0.0)
        actions = self.policy(inputs)
        if noise_scale != 0.0:
            actions += tf.random.normal(shape=actions.shape, stddev=noise_scale)
        actions = tf.clip_by_value(
            actions, self._action_space_low, self._action_space_high
        )
        return actions

    @tf.function
    def _step(self, obs: np.ndarray, **kwargs) -> Dict[str, Union[float, np.ndarray]]:
        if kwargs.get("uniform_sample", False):
            actions = tf.random.uniform(
                [len(obs)], self._action_space_low, self._action_space_high
            )
        else:
            actions = self.call(obs, **kwargs)

        return {SampleBatch.ACTIONS: actions}
