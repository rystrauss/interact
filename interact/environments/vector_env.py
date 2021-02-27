from copy import deepcopy

import gym
import numpy as np
from gym.vector.utils import concatenate


class VectorEnv(gym.vector.SyncVectorEnv):
    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            if info.get("TimeLimit.truncated", False):
                info["TimeLimit.next_obs"] = observation
            if self._dones[i]:
                observation = env.reset()
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )
