from typing import Tuple

import gym
import numpy as np

from interact.common.runners import AbstractRunner


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    ret = 0

    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + gamma * ret * (1. - done)
        discounted.append(ret)

    return discounted[::-1]


class Runner(AbstractRunner):

    def __init__(self, env, policy, nsteps, gamma):
        super().__init__(env, policy, nsteps)

        self.gamma = gamma

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
        ep_infos = []

        for _ in range(self.nsteps):
            # Get actions and value estimates for current observations
            actions, values = self.policy.step(self.obs)
            actions = actions.numpy()
            values = values.numpy()

            # Save data to minibatch buffers
            mb_obs.append(np.copy(self.obs))
            mb_actions.append(actions)
            mb_values.append(values)
            mb_dones.append(self.dones)

            # Clip the actions to avoid out of bound error
            clipped_actions = actions
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

            # Step the environment forward using actions from the policy
            obs, rewards, dones, infos = self.env.step(clipped_actions)

            # Keep track of the number of environment steps the runner has simulated
            self._steps += self.num_env

            for info in infos:
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    ep_infos.append(maybe_ep_info)

            # Save dones and observations for next iteration
            self.dones = dones
            self.obs = obs

            # Save rewards from previous timestep to the minibatch buffer
            mb_rewards.append(rewards)

        mb_dones.append(self.dones)

        # Convert sequence of steps to collection of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype).swapaxes(1, 0).reshape(self.batch_ob_shape)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(0, 1)
        mb_actions = np.asarray(mb_actions, dtype=self.env.action_space.dtype).swapaxes(0, 1)
        mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(0, 1)
        mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(0, 1)
        mb_dones = mb_dones[:, 1:]

        # Get value estimates for the last states
        last_values = self.policy.value(self.obs)

        # Calculate returns; discount and bootstrap off value function
        mb_returns = np.zeros_like(mb_rewards)
        for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
            rewards = rewards.tolist()
            dones = dones.tolist()
            if dones[-1] == 0:
                rewards = discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
            else:
                rewards = discount_with_dones(rewards, dones, self.gamma)

            mb_returns[n] = rewards

        # Convert from [n_env, n_steps, ...] to [n_steps * n_env, ...]
        mb_returns = mb_returns.reshape(-1, *mb_returns.shape[2:])
        mb_actions = mb_actions.reshape(-1, *mb_actions.shape[2:])
        mb_values = mb_values.reshape(-1, *mb_values.shape[2:])

        return mb_obs, mb_returns, mb_actions, mb_values, ep_infos
