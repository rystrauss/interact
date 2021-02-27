from typing import Dict, Callable, Tuple, List

import gin
import gym
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.utils import register
from interact.experience.postprocessing import AdvantagePostprocessor
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.typing import TensorType
from interact.utils.math_utils import explained_variance
from interact.networks import build_network_fn
from interact.policies.actor_critic import ActorCriticPolicy
from interact.schedules import LinearDecay


@gin.configurable(name_or_fn="a2c", blacklist=["env_fn"])
@register("a2c")
class A2CAgent(Agent):
    """The advantage actor-critic algorithm.

    Advantage Actor-Critic (A2C) is a relatively simply actor-critic method which uses
    the advantage function in the policy update.

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
        epsilon: The epsilon value used by the Adam optimizer.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        policy_network: str = "mlp",
        value_network: str = "copy",
        num_envs_per_worker: int = 1,
        num_workers: int = 8,
        use_critic: bool = True,
        use_gae: bool = False,
        lam: float = 1.0,
        gamma: float = 0.99,
        nsteps: int = 5,
        ent_coef: float = 0.01,
        vf_coef: float = 0.25,
        lr: float = 0.0001,
        lr_schedule: str = "constant",
        max_grad_norm: float = 0.5,
        epsilon: float = 1e-7,
    ):
        super().__init__(env_fn)

        assert lr_schedule in {
            "linear",
            "constant",
        }, 'lr_schedule must be "linear" or "constant"'

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
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.max_grad_norm = max_grad_norm
        self.epsilon = epsilon

        self.optimizer = None

    @property
    def timesteps_per_iteration(self):
        return self.nsteps * self.num_envs_per_worker * self.num_workers

    @tf.function
    def _update(self, obs, actions, advantages, returns):
        with tf.GradientTape() as tape:
            # Compute the policy and value predictions for the given observations
            pi, value_preds = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = -pi.log_prob(actions)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the individual loss functions
            policy_loss = tf.reduce_mean(advantages * neglogpacs)
            value_loss = tf.reduce_mean((returns - value_preds) ** 2)
            # The final loss to be minimized is a combination of the policy and value
            # losses, in addition to an entropy bonus which can be used to encourage
            # exploration
            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

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
        }

    @tf.function
    def act(self, obs: TensorType, deterministic: bool = True) -> TensorType:
        pi, _ = self.policy(obs)

        if deterministic:
            actions = pi.mode()
        else:
            actions = pi.mean()

        return actions

    def setup(self, total_timesteps):
        if self.lr_schedule == "linear":
            lr = LinearDecay(self.lr, total_timesteps // self.timesteps_per_iteration)
        else:
            lr = self.lr

        self.optimizer = tf.keras.optimizers.Adam(lr, epsilon=self.epsilon)

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
                self.policy, self.gamma, self.lam, self.use_gae, self.use_critic
            )
        )

        # Aggregate the collected experience so that a gradient update can be performed.
        batch = episodes.to_sample_batch().shuffle()

        # Update the policy and value function based on the new experience.
        metrics = self._update(
            batch[SampleBatch.OBS],
            batch[SampleBatch.ACTIONS],
            batch[SampleBatch.ADVANTAGES],
            batch[SampleBatch.RETURNS],
        )

        return metrics, ep_infos
