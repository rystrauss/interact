from typing import Tuple

import gym
import numpy as np

from interact.agents.base import Agent


def evaluate(
    agent: Agent, env: gym.Env, num_episodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluates an agent's performance when acting deterministically.

    Args:
        agent: The agent to evaluate.
        env: The environment the use for evaluation.
        num_episodes: The number of episodes to complete.

    Returns:
        An array of episodes rewards and an array of episode lengths.
    """
    episodes = 0
    obs = env.reset()

    ep_rewards = []
    ep_lens = []

    while episodes < num_episodes:
        action = agent.act(np.expand_dims(obs, 0), deterministic=True).numpy()
        obs, _, done, info = env.step(action[0])
        if done:
            obs = env.reset()
            episodes += 1

            ep_rewards.append(info["episode"]["reward"])
            ep_lens.append(info["episode"]["length"])

    return np.array(ep_rewards), np.array(ep_lens)
