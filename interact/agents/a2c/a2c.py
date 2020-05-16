from typing import Tuple

from interact.agents.a2c.runner import Runner
from interact.agents.base import Agent
from interact.common.policies import ActorCriticPolicy
from interact.logger import Logger


class A2CAgent(Agent):

    def __init__(self, *, env, load_path=None, policy=None, gamma=0.99, nsteps=5):
        assert isinstance(policy, ActorCriticPolicy), 'policy must be an `ActorCriticPolicy` instance'

        self.policy = policy
        self.gamma = gamma
        self._runner = Runner(env, policy, nsteps, gamma)

        super().__init__(env=env, load_path=load_path)

    def _train_step(self, obs, returns, masks, actions, values) -> Tuple[float, float, float]:
        pass

    def learn(self, *, total_timesteps, logger, log_interval, save_interval):
        assert isinstance(logger, Logger), 'logger must be an instance of the `Logger` class'

        nupdates = total_timesteps // self._runner.batch_size

        for update in range(1, nupdates + 1):
            rollout = self._runner.run()

            policy_loss, value_loss, policy_entropy = self._train_step(*rollout)

    def act(self, observation):
        pass

    def load(self, path):
        self.policy.load_weights(path)

    def save(self, path):
        self.policy.save_weights(path)
