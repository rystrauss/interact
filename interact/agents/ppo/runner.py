"""Provides a `Runner` implementation for PPO.

Author: Ryan Strauss
"""

from typing import Tuple

import gym
import numpy as np

from interact.common.runners import AbstractRunner


class Runner(AbstractRunner):
    """A runner that collects batches of experience for a PPO agent.

    Args:
        env: The Gym environment from which experience will be collected.
        policy: The policy that will be used to collect experience.
        nsteps: The number of steps to be taken in the environment on each call to `run`.
        gamma: The discount factor.
        lam: The lambda term used in the Generalized Advantage Estimation calculation.
    """

    def __init__(self, env, policy, nsteps, gamma, lam):
        super().__init__(env, policy, nsteps)

        self.gamma = gamma
        self.lam = lam

    def run(self):
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [], [], [], [], [], []
        ep_infos = []

        for _ in range(self.nsteps):
            # Get actions and value estimates for current observations
            actions, values, neglogpacs = self.policy.step(self.obs)
            actions = actions.numpy()
            values = values.numpy()
            neglogpacs = neglogpacs.numpy()

            # Save data to minibatch buffers
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # Clip the actions to avoid out of bound error
            clipped_actions = actions
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            # Step the environment forward using actions from the policy
            self.obs[:], rewards, self.dones, infos = self.env.step(clipped_actions)

            # Keep track of the number of environment steps the runner has simulated
            self._steps += self.num_env

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

            # Save rewards from previous timestep to the minibatch buffer
            mb_rewards.append(rewards)

        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)

        # Get value estimates for the last states
        last_values = self.policy.value(self.obs).numpy()

        # Calculate returns with GAE; discount and bootstrap off value function
        mb_advs = np.zeros_like(mb_rewards)
        last_gae_lam = 0

        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                next_nonterminal = 1.0 - self.dones
                next_values = last_values
            else:
                next_nonterminal = 1.0 - mb_dones[t + 1]
                next_values = mb_values[t + 1]

            delta = mb_rewards[t] + self.gamma * next_values * next_nonterminal - mb_values[t]
            mb_advs[t] = last_gae_lam = delta + self.gamma * self.lam * next_nonterminal * last_gae_lam

        mb_returns = mb_advs + mb_values
        return (*map(swap_axes_and_flatten, (mb_obs, mb_returns, mb_actions, mb_values, mb_neglogpacs)), ep_infos)


def swap_axes_and_flatten(x):
    s = x.shape
    return x.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
