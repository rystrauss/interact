import os
from collections import deque
from datetime import datetime
from typing import Optional

import click
import gin
import numpy as np
import ray
import tensorflow as tf
from tqdm import tqdm

from interact.agents.base import Agent
from interact.agents.utils import get_agent
from interact.environments import make_env_fn
from interact.logging import Logger


@gin.configurable
def train(
    agent: str = gin.REQUIRED,
    env_id: str = gin.REQUIRED,
    total_timesteps: int = gin.REQUIRED,
    log_dir: Optional[str] = None,
    log_interval: int = 1,
    save_interval: Optional[int] = None,
    episode_window_size: int = 100,
    verbose=True,
) -> Agent:
    """Trains an agent by repeatedly executing its `train` method.

    Args:
        agent: The type of agent to train.
        env_id: The ID of the environment to train in. Should be a registered Gym
            environment.
        total_timesteps: The total number of environment timesteps for which the agent
            should be trained.
        log_dir: The directory to which the agent and training information should be
            saved.
        log_interval: The frequency, in terms of calls to the agent's `train` method,
            with which logs should be saved.
        save_interval: The frequency, in terms of calls to the agent's `train` method,
            with which model weights should be saved.
        episode_window_size: The number of most recent episodes over which episode
            stats are averaged.
        verbose: A boolean indicating whether or not to display a training progress bar.

    Returns:
        The trained agent.
    """
    total_timesteps = int(total_timesteps)

    if log_dir is None:
        log_dir = os.path.join("logs", env_id, agent)

    log_dir = os.path.join(log_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

    logger = Logger(log_dir)

    agent_name = agent
    agent = get_agent(agent)(make_env_fn(env_id))
    agent.setup(total_timesteps)

    with open(os.path.join(log_dir, "config.gin"), "w") as fp:
        fp.write(gin.operative_config_str())

    checkpoint = tf.train.Checkpoint(agent)
    interval_manager = tf.train.CheckpointManager(
        checkpoint, os.path.join(log_dir, "checkpoints"), max_to_keep=None
    )
    best_manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(log_dir, "best-weights"),
        max_to_keep=1,
        checkpoint_name="best-weights",
    )
    best_reward_mean = -np.inf

    ep_info_buf = deque([], maxlen=episode_window_size)
    metrics = dict()

    update = 0
    pbar = tqdm(total=total_timesteps, desc="Training", disable=not verbose)
    while update * agent.timesteps_per_iteration < total_timesteps:
        batch_metrics, ep_infos = agent.train(update)

        for key, value in batch_metrics.items():
            if key not in metrics:
                metrics[key] = tf.keras.metrics.Mean()

            metrics[key].update_state(value)

        update += 1
        ep_info_buf.extend(ep_infos)

        curr_timesteps = agent.timesteps_per_iteration * update
        if update % log_interval == 0:
            metric_results = {k: v.result() for k, v in metrics.items()}
            logger.log_scalars(curr_timesteps, prefix=agent_name, **metric_results)

            for metric in metrics.values():
                metric.reset_states()

            if ep_info_buf:
                ep_rewards = [ep_info["reward"] for ep_info in ep_info_buf]
                ep_lengths = [ep_info["length"] for ep_info in ep_info_buf]
                scalar_data = {
                    "reward_mean": np.mean(ep_rewards),
                    "reward_min": np.min(ep_rewards),
                    "reward_max": np.max(ep_rewards),
                    "length_mean": np.mean(ep_lengths),
                    "length_min": np.min(ep_lengths),
                    "length_max": np.max(ep_lengths),
                }
                hist_data = {
                    "rewards": ep_rewards,
                    "length": ep_lengths,
                }
                logger.log_scalars(curr_timesteps, prefix="episode", **scalar_data)
                logger.log_histograms(curr_timesteps, prefix="episode", **hist_data)

                if scalar_data["reward_mean"] > best_reward_mean:
                    best_manager.save(curr_timesteps)
                    best_reward_mean = scalar_data["reward_mean"]

        if save_interval is not None and update % save_interval == 0:
            interval_manager.save(curr_timesteps)

        pbar.update(agent.timesteps_per_iteration)

    pbar.close()

    return agent


@click.command("train")
@click.option(
    "--config",
    type=click.Path(dir_okay=False, exists=True),
    nargs=1,
    required=True,
    help="Path to the Gin configuration file to use.",
)
@click.option(
    "--eager",
    is_flag=True,
    help="If flag is set, eager execution will be enabled inside tf.functions.",
)
def main(config, eager):
    """Trains an agent."""
    ray.init()
    gin.parse_config_file(config)
    gin.finalize()
    tf.config.run_functions_eagerly(eager)
    train()


if __name__ == "__main__":
    main()
