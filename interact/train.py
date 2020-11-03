import os
from collections import deque
from datetime import datetime
from typing import List, Callable

import click
import gin
import numpy as np
import ray
import tensorflow as tf
from tqdm import tqdm

from interact.agents.base import Agent
from interact.agents.utils import get_agent
from interact.environments.utils import make_env_fn
from interact.logging import Logger


@gin.configurable
def train(agent: str,
          env_id: str,
          total_timesteps: int,
          log_dir: str = None,
          log_interval: int = 10,
          save_interval: int = 100,
          callbacks: List[Callable] = None,
          verbose=True) -> Agent:
    if log_dir is None:
        log_dir = os.path.join('logs', env_id, agent)

    log_dir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    logger = Logger(log_dir)

    agent = get_agent(agent)(make_env_fn(env_id))

    checkpoint = tf.train.Checkpoint(agent=agent)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    manager = tf.train.CheckpointManager(checkpoint, ckpt_dir, max_to_keep=1)

    ep_info_buf = deque([], maxlen=100)
    metrics = dict()

    update = 0
    pbar = tqdm(total=total_timesteps, desc='Training', disable=not verbose)
    while update * agent.timesteps_per_iteration < total_timesteps:
        batch_metrics, ep_infos = agent.train()

        for key, value in batch_metrics.items():
            if key not in metrics:
                metrics[key] = tf.keras.metrics.Mean()

            metrics[key].update_state(value)

        update += 1
        ep_info_buf.extend(ep_infos)

        if update % log_interval == 0:
            metric_results = {k: v.result() for k, v in metrics.items()}
            logger.log_scalars(update, prefix='agent', **metric_results)
            # TODO: Log total timesteps

            for metric in metrics.values():
                metric.reset_states()

            if ep_info_buf:
                episode_data = {
                    'reward_mean': np.nanmean([ep_info['reward'] for ep_info in ep_info_buf]),
                    'length_mean': np.nanmean([ep_info['length'] for ep_info in ep_info_buf])
                }
                logger.log_scalars(update, prefix='episode', **episode_data)

        if save_interval is not None and update % save_interval == 0:
            manager.save(update)

        if callbacks is not None:
            # TODO: Implement callbacks
            pass

        pbar.update(agent.timesteps_per_iteration)

    pbar.close()

    return agent


@click.command()
@click.option('--config', type=click.Path(dir_okay=False, exists=True), nargs=1, required=True,
              help='Path to the configuration file to use.')
def main(config):
    ray.init()
    gin.parse_config_file(config)
    train()


if __name__ == '__main__':
    main()
