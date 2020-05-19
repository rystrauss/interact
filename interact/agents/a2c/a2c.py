from typing import Tuple

import tensorflow as tf
from tqdm import tqdm

from interact.agents.a2c.runner import Runner
from interact.agents.base import Agent
from interact.common.policies import ActorCriticPolicy
from interact.logger import Logger


class A2CAgent(Agent):

    def __init__(self, *, env, policy=None, load_path=None, gamma=0.99, nsteps=5, ent_coef=0.01, vf_coef=0.25,
                 learning_rate=0.0001):
        assert isinstance(policy, ActorCriticPolicy), 'policy must be an `ActorCriticPolicy` instance'

        self.policy = policy
        self.gamma = gamma
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self._runner = Runner(env, policy, nsteps, gamma)
        self._optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

        super().__init__(env=env, load_path=load_path)

    def _train_step(self, obs, returns, actions, values) -> Tuple[float, float, float]:
        advantages = returns - values

        with tf.GradientTape() as tape:
            pi = self.policy(obs)
            neglogpacs = -pi.log_prob(actions)
            entropy = pi.entropy()
            policy_loss = tf.reduce_mean(advantages * neglogpacs)
            value_loss = tf.reduce_mean((returns - tf.squeeze(self.policy.value(obs))) ** 2)
            loss = policy_loss - entropy * self.ent_coef + value_loss * self.vf_coef

        grads = tape.gradient(loss, self.policy.trainable_weights)
        self._optimizer.apply_gradients(zip(grads, self.policy.trainable_weights))

        return policy_loss, value_loss, entropy

    def learn(self, *, total_timesteps, logger, log_interval, save_interval):
        assert isinstance(logger, Logger), 'logger must be an instance of the `Logger` class'

        nupdates = total_timesteps // self._runner.batch_size

        for update in tqdm(range(1, nupdates + 1), desc='Updates'):
            rollout = self._runner.run()
            policy_loss, value_loss, entropy = self._train_step(*rollout)

    def act(self, observation):
        pass

    def load(self, path):
        self.policy.load_weights(path)

    def save(self, path):
        self.policy.save_weights(path)
