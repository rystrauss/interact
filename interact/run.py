"""Primary command line utility for training and playing with agents.

Author: Ryan Strauss
"""

import json
import multiprocessing
import os
from datetime import datetime

import click
import gym
import numpy as np
import tensorflow as tf

from interact.agents.util import available_agents, get_agent
from interact.common.parallel_env import make_parallelized_env
from interact.logger import Logger, printc, Colors


def extract_extra_kwargs(context_args):
    def pairwise(iterable):
        iterator = iter(iterable)
        return zip(iterator, iterator)

    def parse(v):
        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    kwargs = {k.lstrip('--'): parse(v) for k, v in pairwise(context_args)}
    return kwargs


@click.group()
def cli():
    """This command line interface allows agents to be trained and visualized."""
    pass


@cli.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True
))
@click.option('--env', type=click.STRING, nargs=1, help='The name of the Gym environment to train on.', required=True)
@click.option('--agent', type=click.Choice(available_agents()), nargs=1, help='The agent to use for learning.',
              required=True)
@click.option('--total_timesteps', type=click.INT, nargs=1, default=100000,
              help='The total number of steps in the environment to train for.')
@click.option('--log_interval', type=click.INT, nargs=1, default=100,
              help='The frequency, in terms of policy updates, that logs will be saved.')
@click.option('--save_interval', type=click.INT, nargs=1, default=500,
              help='The frequency, in terms of policy updates, that model weights will be saved.')
@click.option('--num_env', type=click.INT, nargs=1, default=-1,
              help='The number of environments to be simulated in parallel. '
                   'If -1, this will be set to the number of available CPUs.')
@click.option('--seed', type=click.INT, nargs=1, help='The random seed to be used.')
@click.option('--log_dir', type=click.Path(), nargs=1, help='Directory where experiment data will be saved.')
@click.option('--load_path', type=click.Path(), nargs=1,
              help='Path to network weights which be loaded before training.')
@click.pass_context
def train(context, env, agent, total_timesteps, log_interval, save_interval, num_env, seed, log_dir, load_path):
    """Executes the training process of an agent."""
    if num_env == -1:
        num_env = multiprocessing.cpu_count()

    if seed is None:
        seed = int.from_bytes(os.urandom(4), byteorder='big')

    tf.random.set_seed(seed)

    if log_dir is None:
        log_dir = os.path.join('logs', env)

    extra_kwargs = extract_extra_kwargs(context.args)

    log_dir = os.path.join(log_dir, datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    logger = Logger(log_dir)

    with open(os.path.join(log_dir, 'params.json'), 'w') as fp:
        json.dump(dict(env=env, agent=agent, **extra_kwargs), fp)

    env = make_parallelized_env(env, num_env, seed)

    agent = get_agent(agent)(env=env, load_path=load_path, **extra_kwargs)

    agent.learn(total_timesteps=total_timesteps,
                logger=logger,
                log_interval=log_interval,
                save_interval=save_interval)

    logger.close()
    env.close()


@cli.command()
@click.option('--dir', type=click.Path(exists=True, file_okay=False), nargs=1,
              help='Path to directory of the agent being loaded.')
def play(dir):
    """Visualizes a trained agent playing in its environment."""
    with open(os.path.join(dir, 'params.json'), 'r') as fp:
        params = json.load(fp)

    env = params['env']
    agent = params['agent']
    del params['env']
    del params['agent']

    load_path = tf.train.latest_checkpoint(os.path.join(dir, 'weights'))

    env = gym.make(env)
    agent = get_agent(agent)(env=env, load_path=load_path, **params)

    obs = env.reset()

    try:
        while True:
            action = agent.act(np.expand_dims(obs, axis=0))
            obs, _, done, _ = env.step(np.squeeze(action))
            env.render()
            if done:
                obs = env.reset()
    except KeyboardInterrupt:
        printc(Colors.BLUE, 'Got KeyboardInterrupt: exiting...')
    finally:
        env.close()


if __name__ == '__main__':
    cli()
