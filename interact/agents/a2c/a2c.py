from typing import Dict, Callable

import gym

from interact.agents.base import Agent


class A2CAgent(Agent):

    def __init__(self,
                 env_fn: Callable[[], gym.Env],
                 policy_network: str = 'mlp',
                 value_network: str = 'copy',
                 num_envs_per_worker: int = 1,
                 num_workers: int = 1,
                 gamma: float = 0.99,
                 nsteps: int = 5,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.25,
                 learning_rate: float = 0.0001,
                 lr_schedule: str = 'constant',
                 max_grad_norm: float = 0.5,
                 rho: float = 0.99,
                 epsilon: float = 1e-5):
        super().__init__(env_fn)

    @property
    def timesteps_per_iteration(self):
        pass

    def train(self) -> Dict[str, float]:
        pass

    def act(self, obs, state=None):
        pass
