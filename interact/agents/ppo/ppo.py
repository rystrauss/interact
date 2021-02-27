from typing import Tuple, Dict, List, Callable

import gin
import gym
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.utils import register
from interact.experience.postprocessing import AdvantagePostprocessor
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.networks import build_network_fn
from interact.policies.actor_critic import ActorCriticPolicy
from interact.schedules import LinearDecay
from interact.typing import TensorType
from interact.utils.math_utils import explained_variance


@gin.configurable(name_or_fn="ppo", blacklist=["env_fn"])
@register("ppo")
class PPOAgent(Agent):
    """The Proximal Policy Optimization algorithm.

    This implementation uses the clipped surrogate objective.

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
        noptepochs: Number of epochs over each batch when optimizing the loss.
        cliprange: Clipping parameter used in the surrogate loss.
        cliprange_schedule: The schedule for the clipping parameter, either 'constant'
            or 'linear'.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        policy_network: str = "mlp",
        value_network: str = "copy",
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
        noptepochs: int = 4,
        cliprange: float = 0.2,
        cliprange_schedule: str = "constant",
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

        network_fn = build_network_fn(policy_network, env.observation_space.shape)

        def policy_fn():
            return ActorCriticPolicy(
                env.observation_space, env.action_space, network_fn, value_network
            )

        self.policy = policy_fn()
        self.runner = Runner(
            env_fn,
            policy_fn,
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
        self.noptepochs = noptepochs
        self.cliprange = cliprange
        self.cliprange_schedule = cliprange_schedule

        self.optimizer = None

    @property
    def timesteps_per_iteration(self) -> int:
        return self.nsteps * self.num_envs_per_worker * self.num_workers

    @tf.function
    def _update(self, obs, actions, advantages, returns, old_neglogpacs, cliprange):
        # Normalize the advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (
            tf.math.reduce_std(advantages) + 1e-8
        )

        with tf.GradientTape() as tape:
            # Compute the policy and value predictions for the given observations
            pi, value_preds = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = -pi.log_prob(actions)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the policy surrogate loss
            ratio = tf.exp(old_neglogpacs - neglogpacs)
            pg_loss_unclipped = -advantages * ratio
            pg_loss_clipped = -advantages * tf.clip_by_value(
                ratio, 1 - cliprange, 1 + cliprange
            )
            policy_loss = tf.reduce_mean(tf.maximum(pg_loss_unclipped, pg_loss_clipped))
            # Define the value loss
            value_preds_clipped = tf.clip_by_value(
                value_preds, -self.vf_clip, self.vf_clip
            )
            vf_loss_unclipped = (returns - value_preds) ** 2
            vf_loss_clipped = (returns - value_preds_clipped) ** 2
            value_loss = 0.5 * tf.reduce_mean(
                tf.maximum(vf_loss_clipped, vf_loss_unclipped)
            )
            # The final loss to be minimized is a combination of the policy and value
            # losses, in addition to an entropy bonus which can be used to encourage
            # exploration
            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

        clipfrac = tf.reduce_mean(
            tf.cast(tf.greater(tf.abs(ratio - 1.0), cliprange), tf.float32)
        )

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.trainable_weights)
        # Perform gradient clipping
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        # This is a measure of how well the value function explains the variance in
        # the rewards
        value_explained_variance = explained_variance(returns, value_preds)

        return {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "policy_entropy": entropy,
            "value_explained_variance": value_explained_variance,
            "clipfrac": clipfrac,
        }

    def act(self, obs: TensorType, deterministic: bool = True) -> TensorType:
        pi, _ = self.policy(obs)

        if deterministic:
            actions = pi.mode()
        else:
            actions = pi.mean()

        return actions

    def train(self, update: int) -> Tuple[Dict[str, float], List[Dict]]:
        # Update the weights of the actor policies to be consistent with the most
        # recent update.
        self.runner.update_policies(self.policy.get_weights())

        # Rollout the current policy in the environment to get back a batch of
        # experience.
        episodes, ep_infos = self.runner.run(self.nsteps)

        # Compute advantages for the collected experience.
        episodes.for_each(
            AdvantagePostprocessor(
                self.policy.value, self.gamma, self.lam, self.use_gae, self.use_critic
            )
        )

        # Aggregate the collected experience so that a gradient update can be performed.
        batch = episodes.to_sample_batch()

        metric_means = dict()

        curr_cliprange = (
            self.cliprange
            if self.cliprange_schedule == "constant"
            else self.cliprange(update * self.timesteps_per_iteration)
        )

        for _ in range(self.noptepochs):
            batch.shuffle()
            for minibatch in batch.to_minibatches(self.nminibatches):
                # Update the policy and value function based on the new experience.
                metrics = self._update(
                    minibatch[SampleBatch.OBS],
                    minibatch[SampleBatch.ACTIONS],
                    minibatch[SampleBatch.ADVANTAGES],
                    minibatch[SampleBatch.RETURNS],
                    -minibatch[SampleBatch.ACTION_LOGP],
                    curr_cliprange,
                )

                for k, v in metrics.items():
                    if k not in metric_means:
                        metric_means[k] = tf.keras.metrics.Mean()

                    metric_means[k].update_state(v)

        metrics = {k: v.result() for k, v in metric_means.items()}
        return metrics, ep_infos

    def setup(self, total_timesteps):
        if self.lr_schedule == "linear":
            lr = LinearDecay(self.lr, total_timesteps // self.timesteps_per_iteration)
        else:
            lr = self.lr

        self.optimizer = tf.keras.optimizers.Adam(lr)

        if self.cliprange_schedule == "linear":
            self.cliprange = LinearDecay(self.cliprange, total_timesteps)
