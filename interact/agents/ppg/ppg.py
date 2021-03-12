from typing import Callable, Tuple, Dict, List

import gin
import gym
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from interact.agents.base import Agent
from interact.agents.ppg.policy import PPGPolicy
from interact.agents.utils import register
from interact.experience.episode_batch import EpisodeBatch
from interact.experience.postprocessing import AdvantagePostprocessor
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.schedules import LinearDecay
from interact.typing import TensorType
from interact.utils.math import explained_variance


@gin.configurable(name_or_fn="ppg", denylist=["env_fn"])
@register("ppg")
class PPGAgent(Agent):
    """The Phasic Policy Gradients algorithm.

    Args:
        env_fn: A function that, when called, returns an instance of the agent's
            environment.
        policy_network: The type of model to use for the policy network.
        value_network: Either 'copy' or 'shared', indicating whether or not weights
            should be shared between the policy and value networks.
        num_envs_per_worker: The number of synchronous environments to be executed in
            each worker.
        num_workers: The number of parallel workers to use for experience collection.
        use_critic: Whether to use critic (value estimates). Setting this to False will
            use 0 as baseline. If this is false, the agent becomes a vanilla
            actor-critic method.
        use_gae: Whether or not to use GAE.
        lam: The lambda parameter used in GAE.
        gamma: The discount factor.
        nsteps: The number of steps taken in each environment per update.
        ent_coef: The coefficient of the entropy term in the loss function.
        vf_coef: The coefficient of the value term in the loss function.
        lr: The initial learning rate.
        lr_schedule: The schedule for the learning rate, either 'constant' or 'linear'.
        max_grad_norm: The maximum value for the gradient clipping.
        nminibatches: Number of training minibatches per update.
        cliprange: Clipping parameter used in the surrogate loss.
        cliprange_schedule: The schedule for the clipping parameter, either 'constant'
            or 'linear'.
        policy_iterations: The number of policy updates performed in each policy phase.
        policy_epochs: Controls the sample reuse for the policy function.
        value_epochs: Controls the sample reuse for the value function.
        auxiliary_epochs: Controls the sample reuse during the auxiliary phase,
            representing the number of epochs performed across all data in the replay
            buffer.
        bc_coef: Coefficient for the behavior cloning component of the joint loss.
        nminibatches_aux: Number of training minibatches per auxiliary epoch.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        policy_network: str = "mlp",
        num_envs_per_worker: int = 1,
        num_workers: int = 8,
        use_critic: bool = True,
        use_gae: bool = True,
        lam: float = 0.95,
        gamma: float = 0.99,
        nsteps: int = 128,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        vf_clip: float = 10.0,
        lr: float = 2.5e-4,
        lr_schedule: str = "constant",
        max_grad_norm: float = 0.5,
        nminibatches: int = 4,
        cliprange: float = 0.2,
        cliprange_schedule: str = "constant",
        policy_iterations: int = 32,
        policy_epochs: int = 1,
        value_epochs: int = 1,
        auxiliary_epochs: int = 6,
        bc_coef: float = 1.0,
        nminibatches_aux: int = 16,
    ):
        super().__init__(env_fn)

        assert lr_schedule in {
            "linear",
            "constant",
        }, 'lr_schedule must be "linear" or "constant"'
        assert cliprange_schedule in {
            "linear",
            "constant",
        }, 'cliprange_schedule must be "linear" or "constant"'

        env = self.make_env()

        def policy_fn():
            return PPGPolicy(env.observation_space, env.action_space, policy_network)

        self.policy = policy_fn()
        self.policy.build([None, *env.observation_space.shape])

        self.runner = None
        self.runner_config = dict(
            env_fn=env_fn,
            policy_fn=policy_fn,
            num_envs_per_worker=num_envs_per_worker,
            num_workers=num_workers,
        )

        self.num_envs_per_worker = num_envs_per_worker
        self.num_workers = num_workers
        self.use_critic = use_critic
        self.use_gae = use_gae
        self.lam = lam
        self.gamma = gamma
        self.nsteps = nsteps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.vf_clip = vf_clip
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.max_grad_norm = max_grad_norm
        self.nminibatches = nminibatches
        self.cliprange = cliprange
        self.cliprange_schedule = cliprange_schedule
        self.policy_iterations = policy_iterations
        self.policy_epochs = policy_epochs
        self.value_epochs = value_epochs
        self.auxiliary_epochs = auxiliary_epochs
        self.bc_coef = bc_coef
        self.nminibatches_aux = nminibatches_aux

        self.policy_optimizer = None
        self.value_optimizer = None
        self.aux_optimizer = None

    @property
    def timesteps_per_iteration(self) -> int:
        return (
            self.nsteps
            * self.num_envs_per_worker
            * self.num_workers
            * self.policy_iterations
        )

    @tf.function
    def _train_policy(self, obs, actions, advantages, old_neglogpacs, cliprange):
        with tf.GradientTape() as tape:
            # Compute the policy for the given observations
            pi = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = -pi.log_prob(actions)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the policy surrogate loss as per PPO
            ratio = tf.exp(old_neglogpacs - neglogpacs)
            pg_loss_unclipped = -advantages * ratio
            pg_loss_clipped = -advantages * tf.clip_by_value(
                ratio, 1 - cliprange, 1 + cliprange
            )
            policy_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
            loss = policy_loss - entropy * self.ent_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.policy_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.policy_optimizer.apply_gradients(zip(grads, self.policy.policy_weights))

        return policy_loss, entropy

    @tf.function
    def _train_value(self, obs, returns):
        with tf.GradientTape() as tape:
            value_preds = self.policy.value(obs)
            value_loss = 0.5 * tf.reduce_mean((returns - value_preds) ** 2)
            loss = value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.value_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.value_optimizer.apply_gradients(zip(grads, self.policy.value_weights))

        return value_loss

    @tf.function
    def _train_policy_and_value(
        self, obs, actions, advantages, returns, old_neglogpacs, cliprange
    ):
        with tf.GradientTape() as tape:
            # Compute the policy for the given observations
            pi = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = -pi.log_prob(actions)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the policy surrogate loss as per PPO
            ratio = tf.exp(old_neglogpacs - neglogpacs)
            pg_loss_unclipped = -advantages * ratio
            pg_loss_clipped = -advantages * tf.clip_by_value(
                ratio, 1 - cliprange, 1 + cliprange
            )
            policy_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))

            value_preds = self.policy.value(obs)
            value_loss = 0.5 * tf.reduce_mean((returns - value_preds) ** 2)

            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.policy_and_value_weights)
        # Perform gradient clipping
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.policy_optimizer.apply_gradients(
            zip(grads, self.policy.policy_and_value_weights)
        )

        return policy_loss, entropy, value_loss

    @tf.function
    def _train_auxiliary(self, obs, returns, old_pi_logits):
        if self.policy.is_discrete:
            old_pi = tfd.Categorical(old_pi_logits)
        else:
            mean, logstd = tf.split(old_pi_logits, 2, axis=-1)
            old_pi = tfd.MultivariateNormalDiag(mean, tf.exp(logstd))

        with tf.GradientTape() as tape:
            pi, aux_value_preds = self.policy.auxiliary_heads(obs)

            bc_loss = tf.reduce_mean(old_pi.kl_divergence(pi))
            aux_value_loss = 0.5 * tf.reduce_mean((returns - aux_value_preds) ** 2)

            joint_loss = aux_value_loss + self.bc_coef * bc_loss

        grads = tape.gradient(joint_loss, self.policy.auxiliary_weights)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        self.aux_optimizer.apply_gradients(zip(grads, self.policy.auxiliary_weights))

        with tf.GradientTape() as tape:
            value_preds = self.policy.value(obs)
            value_loss = 0.5 * tf.reduce_mean((returns - value_preds) ** 2)

        grads = tape.gradient(value_loss, self.policy.value_weights)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)

        if self.value_optimizer is not None:
            self.value_optimizer.apply_gradients(zip(grads, self.policy.value_weights))
        else:
            self.policy_optimizer.apply_gradients(zip(grads, self.policy.value_weights))

    def _policy_phase(self, curr_cliprange):
        ep_info_buffer = []
        metric_means = dict()

        experience_buffer = []

        for _ in range(self.policy_iterations):
            self.runner.update_policies(self.policy.get_weights())
            episodes, ep_infos = self.runner.run(self.nsteps)
            ep_info_buffer.extend(ep_infos)

            episodes.for_each(
                AdvantagePostprocessor(
                    self.policy.value,
                    self.gamma,
                    self.lam,
                    self.use_gae,
                    self.use_critic,
                )
            )
            experience_buffer.append(episodes)
            batch = episodes.to_sample_batch()

            if self.policy_epochs == self.value_epochs:
                for _ in range(self.policy_epochs):
                    batch.shuffle()
                    for minibatch in batch.to_minibatches(self.nminibatches):
                        policy_loss, entropy, value_loss = self._train_policy_and_value(
                            minibatch[SampleBatch.OBS],
                            minibatch[SampleBatch.ACTIONS],
                            minibatch[SampleBatch.ADVANTAGES],
                            minibatch[SampleBatch.RETURNS],
                            -minibatch[SampleBatch.ACTION_LOGP],
                            curr_cliprange,
                        )

                        value_explained_variance = explained_variance(
                            tf.constant(minibatch[SampleBatch.RETURNS]),
                            tf.constant(minibatch[SampleBatch.VALUE_PREDS]),
                        )

                        metrics = {
                            "policy_loss": policy_loss,
                            "policy_entropy": entropy,
                            "value_loss": value_loss,
                            "value_explained_variance": value_explained_variance,
                        }

                        for k, v in metrics.items():
                            if k not in metric_means:
                                metric_means[k] = tf.keras.metrics.Mean()

                            metric_means[k].update_state(v)
            else:
                for _ in range(self.policy_epochs):
                    batch.shuffle()
                    for minibatch in batch.to_minibatches(self.nminibatches):
                        policy_loss, entropy = self._train_policy(
                            minibatch[SampleBatch.OBS],
                            minibatch[SampleBatch.ACTIONS],
                            minibatch[SampleBatch.ADVANTAGES],
                            -minibatch[SampleBatch.ACTION_LOGP],
                            curr_cliprange,
                        )

                        metrics = {
                            "policy_loss": policy_loss,
                            "policy_entropy": entropy,
                        }

                        for k, v in metrics.items():
                            if k not in metric_means:
                                metric_means[k] = tf.keras.metrics.Mean()

                            metric_means[k].update_state(v)

                for _ in range(self.value_epochs):
                    batch.shuffle()
                    for minibatch in batch.to_minibatches(self.nminibatches):
                        value_loss = self._train_value(
                            minibatch[SampleBatch.OBS], minibatch[SampleBatch.RETURNS]
                        )

                        value_explained_variance = explained_variance(
                            tf.constant(minibatch[SampleBatch.RETURNS]),
                            tf.constant(minibatch[SampleBatch.VALUE_PREDS]),
                        )
                        metrics = {
                            "value_loss": value_loss,
                            "value_explained_variance": value_explained_variance,
                        }

                        for k, v in metrics.items():
                            if k not in metric_means:
                                metric_means[k] = tf.keras.metrics.Mean()

                            metric_means[k].update_state(v)

        metrics = {k: v.result() for k, v in metric_means.items()}

        return metrics, EpisodeBatch.merge(experience_buffer), ep_info_buffer

    def _auxiliary_phase(self, batch: SampleBatch):
        pi_logits = []

        for minibatch in batch.to_minibatches(self.nminibatches_aux):
            pi_logits.append(self.policy.policy_logits(minibatch[SampleBatch.OBS]))

        batch[SampleBatch.POLICY_LOGITS] = np.vstack(pi_logits)

        for _ in range(self.auxiliary_epochs):
            batch.shuffle()
            for minibatch in batch.to_minibatches(self.nminibatches_aux):
                self._train_auxiliary(
                    minibatch[SampleBatch.OBS],
                    minibatch[SampleBatch.RETURNS],
                    minibatch[SampleBatch.POLICY_LOGITS],
                )

    @tf.function
    def act(self, obs: TensorType, deterministic: bool = True) -> TensorType:
        pi, _ = self.policy(obs)

        if deterministic:
            actions = pi.mode()
        else:
            actions = pi.mean()

        return actions

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        curr_cliprange = (
            self.cliprange
            if self.cliprange_schedule == "constant"
            else self.cliprange(update * self.timesteps_per_iteration)
        )

        metrics, episodes, ep_infos = self._policy_phase(curr_cliprange)

        batch = episodes.to_sample_batch()

        self._auxiliary_phase(batch)

        return metrics, ep_infos

    def pretrain_setup(self, total_timesteps: int):
        if self.lr_schedule == "linear":
            lr = LinearDecay(self.lr, total_timesteps // self.timesteps_per_iteration)
        else:
            lr = self.lr

        if self.cliprange_schedule == "linear":
            self.cliprange = LinearDecay(self.cliprange, total_timesteps)

        self.policy_optimizer = tf.optimizers.Adam(learning_rate=lr)
        if self.policy_epochs != self.value_epochs:
            self.value_optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.aux_optimizer = tf.optimizers.Adam(learning_rate=lr)

        self.runner = Runner(**self.runner_config)
