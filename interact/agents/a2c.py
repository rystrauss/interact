from typing import Dict, Callable, Tuple, List

import gin
import gym
import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.utils import register
from interact.experience.postprocessing import AdvantagePostprocessor
from interact.experience.runner import Runner
from interact.experience.sample_batch import SampleBatch
from interact.math_utils import explained_variance
from interact.networks import build_network_fn
from interact.policies.actor_critic import ActorCriticPolicy
from interact.schedules import LinearDecay


@gin.configurable(name_or_fn='a2c', blacklist=['env_fn'])
@register('a2c')
class A2CAgent(Agent):

    def __init__(self,
                 env_fn: Callable[[], gym.Env],
                 policy_network: str = 'mlp',
                 value_network: str = 'copy',
                 num_envs_per_worker: int = 1,
                 num_workers: int = 1,
                 use_critic: bool = True,
                 use_gae: bool = True,
                 lam: float = 1.0,
                 gamma: float = 0.99,
                 nsteps: int = 20,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 lr: float = 0.0001,
                 lr_schedule: str = 'constant',
                 max_grad_norm: float = 0.5):
        super().__init__(env_fn)

        assert lr_schedule in {'linear', 'constant'}, 'lr_schedule must be "linear" or "constant"'

        env = self.make_env()

        network_fn = build_network_fn(policy_network, env.observation_space.shape)

        def policy_fn():
            return ActorCriticPolicy(env.observation_space, env.action_space, network_fn, value_network)

        self.policy = policy_fn()
        self.runner = Runner(env_fn, policy_fn, num_envs_per_worker=num_envs_per_worker, num_workers=num_workers)

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

        self.optimizer = None

    @property
    def timesteps_per_iteration(self):
        return self.nsteps * self.num_envs_per_worker * self.num_workers

    @tf.function
    def _update(self, obs, actions, advantages, returns):
        with tf.GradientTape() as tape:
            # Compute the policy for the given observations
            pi, value_preds = self.policy(obs)
            # Retrieve policy entropy and the negative log probabilities of the actions
            neglogpacs = -pi.log_prob(actions)
            entropy = tf.reduce_mean(pi.entropy())
            # Define the individual loss functions
            policy_loss = tf.reduce_mean(advantages * neglogpacs)
            value_loss = tf.reduce_mean((returns - value_preds) ** 2)
            # The final loss to be minimized is a combination of the policy and value losses, in addition
            # to an entropy bonus which can be used to encourage exploration
            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

        # Perform a gradient update to minimize the loss
        grads = tape.gradient(loss, self.policy.trainable_weights)
        # Perform gradient clipping
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        # Apply the gradient update
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        value_explained_variance = explained_variance(returns, value_preds)

        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'policy_entropy': entropy,
            'value_explained_variance': value_explained_variance
        }

    def act(self, obs, state=None):
        pi, _ = self.policy(obs)
        return pi.mode()

    def setup(self, total_timesteps):
        if self.lr_schedule == 'linear':
            lr = LinearDecay(self.lr, total_timesteps // self.timesteps_per_iteration)
        else:
            lr = self.lr

        self.optimizer = tf.keras.optimizers.Adam(lr)

    def train(self) -> Tuple[Dict[str, float], List[Dict]]:
        self.runner.update_policies(self.policy.get_weights())

        episodes, ep_infos = self.runner.run(self.nsteps)

        episodes.for_each(AdvantagePostprocessor(self.policy, self.gamma, self.lam, self.use_gae, self.use_critic))

        batch = episodes.to_sample_batch().shuffle()

        metrics = self._update(batch[SampleBatch.OBS],
                               batch[SampleBatch.ACTIONS],
                               batch[SampleBatch.ADVANTAGES],
                               batch[SampleBatch.RETURNS])

        return metrics, ep_infos