"""Primary command line utility for training and playing with agents.

Author: Ryan Strauss
"""

import json
import multiprocessing
import os
from datetime import datetime

import click
import tensorflow as tf

from interact.agents.util import available_agents, get_agent
from interact.logger import Logger, printc, Colors
from interact.run_util import extract_extra_kwargs, make_parallelized_env


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
@click.option('--normalize_obs', is_flag=True)
@click.option('--normalize_rewards', is_flag=True)
@click.pass_context
def train(context, env, agent, total_timesteps, log_interval, save_interval, num_env, seed, log_dir, load_path,
          normalize_obs, normalize_rewards):
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
        params = dict(env=env,
                      agent=agent,
                      total_timesteps=total_timesteps,
                      num_env=num_env,
                      seed=seed,
                      normalize_obs=normalize_obs,
                      normalize_rewards=normalize_rewards,
                      load_path=load_path,
                      **extra_kwargs)
        json.dump(params, fp)

    env = make_parallelized_env(env, num_env, seed, normalize_obs, normalize_rewards)

    agent = get_agent(agent)(env=env, load_path=load_path, **extra_kwargs)

    agent.learn(total_timesteps=total_timesteps,
                logger=logger,
                log_interval=log_interval,
                save_interval=save_interval)

    logger.close()
    env.close()


def load_agent_and_env(path, monitor=False):
    """Loads a pretrained agent and its corresponding environment.

    Args:
        path: Path to the agent's directory (which was created by the training process).
        monitor: If True, the environment will be loaded with a video recorded attached.

    Returns:
        A 2-tuple with:
            agent: the loaded agent
            env: the loaded environment
    """
    with open(os.path.join(path, 'params.json'), 'r') as fp:
        params = json.load(fp)

    env = params['env']
    agent = params['agent']
    normalize_obs = params['normalize_obs']
    normalize_rewards = params['normalize_rewards']
    del params['env']
    del params['agent']
    del params['total_timesteps']
    del params['num_env']
    del params['seed']
    del params['normalize_obs']
    del params['normalize_rewards']
    del params['load_path']

    load_path = tf.train.latest_checkpoint(os.path.join(path, 'weights'))

    if monitor:
        env = make_parallelized_env(env, 1, normalize_obs=normalize_obs, normalize_rewards=normalize_rewards,
                                    video_callable=lambda x: x != 0, video_path=os.path.join(path, 'video'))
    else:
        env = make_parallelized_env(env, 1, normalize_obs=normalize_obs, normalize_rewards=normalize_rewards)

    agent = get_agent(agent)(env=env, load_path=load_path, **params)

    return agent, env


@cli.command()
@click.option('--dir', type=click.Path(exists=True, file_okay=False), nargs=1,
              help='Path to directory of the agent being loaded.')
@click.option('--save', is_flag=True, help='If flag is set, episode recordings will be saved.')
def play(dir, save):
    """Visualizes (and optionally saves) a trained agent acting in its environment."""
    agent, env = load_agent_and_env(dir, monitor=save)

    obs = env.reset()

    try:
        while True:
            action = agent.act(obs).numpy()
            obs, _, done, _ = env.step(action)
            env.render()
    except KeyboardInterrupt:
        printc(Colors.BLUE, 'Got KeyboardInterrupt: exiting...')
    finally:
        env.close()


if __name__ == '__main__':
    cli()
