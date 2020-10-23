import os
from collections import deque
from typing import List, Callable

import tensorflow as tf
from tqdm import tqdm

from interact.agents.base import Agent
from interact.agents.utils import get_agent
from interact.environments.utils import make_env_fn
from interact.logging import Logger


def train(agent: str,
          env_id: str,
          total_timesteps: int,
          log_dir: str,
          log_interval: int = 1,
          save_interval: int = None,
          callbacks: List[Callable] = None,
          verbose=True) -> Agent:
    agent = get_agent(agent)(make_env_fn(env_id))

    logger = Logger(log_dir)

    checkpoint = tf.train.Checkpoint(agent=agent)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=1, checkpoint_interval=save_interval)

    ep_info_buf = deque([], maxlen=100)

    updates = 0
    pbar = tqdm(total=total_timesteps, desc='Timesteps', disable=not verbose)
    while updates * agent.timesteps_per_iteration < total_timesteps:
        metrics, ep_infos = agent.train()

        updates += 1
        ep_info_buf.extend(ep_infos)

        if updates % log_interval == 0:
            logger.log_scalars(updates, **metrics)

        if save_interval is not None:
            manager.save(updates)

        if callbacks is not None:
            # TODO: Implement callbacks
            pass

        pbar.update(agent.timesteps_per_iteration)

    pbar.close()

    return agent
