import os
from typing import List, Callable

import tensorflow as tf

from interact.agents.base import Agent
from interact.agents.utils import get_agent


def train(agent: str,
          env: str,
          total_timesteps: int,
          log_dir: str,
          log_interval: int = 1,
          save_interval: int = None,
          callbacks: List[Callable] = None) -> Agent:
    agent = get_agent(agent)(env=env)

    checkpoint = tf.train.Checkpoint(agent=agent)
    ckpt_prefix = os.path.join(log_dir, 'checkpoints/weights')

    updates = 0
    while updates * agent.timesteps_per_iteration < total_timesteps:
        agent.train()
        updates += 1

        if updates % log_interval == 0:
            pass

        if updates % save_interval == 0:
            checkpoint.save(ckpt_prefix)

        if callbacks is not None:
            pass

    return agent
