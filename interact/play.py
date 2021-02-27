import os

import click
import gin
import numpy as np
import tensorflow as tf
from gym.wrappers import Monitor

from interact.agents.utils import get_agent
from interact.environments import make_env_fn


def play(agent_dir, num_episodes, max_episode_steps, save_videos):
    agent = get_agent(gin.query_parameter("train.agent"))(
        make_env_fn(
            gin.query_parameter("train.env_id"), episode_time_limit=max_episode_steps
        )
    )
    agent.setup(gin.query_parameter("train.total_timesteps"))

    ckpt_path = tf.train.latest_checkpoint(os.path.join(agent_dir, "checkpoints"))
    checkpoint = tf.train.Checkpoint(agent)
    checkpoint.restore(ckpt_path).expect_partial()

    env = agent.make_env()

    if save_videos:
        env = Monitor(
            env,
            os.path.join(agent_dir, "monitor"),
            video_callable=lambda _: True,
            force=True,
        )

    try:
        episodes = 0
        obs = env.reset()
        while episodes < num_episodes:
            action = agent.act(np.expand_dims(obs, 0), deterministic=True).numpy()
            obs, _, done, _ = env.step(action[0])
            env.render()
            if done:
                obs = env.reset()
                episodes += 1
    except KeyboardInterrupt:
        env.close()


@click.command("play")
@click.option(
    "--agent_dir",
    type=click.Path(file_okay=False, exists=True),
    nargs=1,
    required=True,
    help="Path to the directory that contains the agent you wish to visualize.",
)
@click.option(
    "--num_episodes",
    type=click.INT,
    nargs=1,
    default=np.iinfo(np.int64).max,
    help="The number of episodes to execute.",
)
@click.option(
    "--max_episode_steps",
    type=click.INT,
    nargs=1,
    help="The maximum length of an episode. If not specified, the default for the "
    "given environment is used.",
)
@click.option(
    "--save_videos", is_flag=True, help="If flag is set, episode videos will be saved."
)
def main(agent_dir, num_episodes, max_episode_steps, save_videos):
    """Visualizes an agent acting in its environment."""
    # noinspection PyUnresolvedReferences
    from interact import train

    gin.parse_config_file(os.path.join(agent_dir, "config.gin"))
    gin.finalize()
    play(agent_dir, num_episodes, max_episode_steps, save_videos)


if __name__ == "__main__":
    main()
