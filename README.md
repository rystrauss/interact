# Interact

Interact contains implementations of several deep reinforcement learning algorithms.

## Installation

Interact can be installed as follows:

```bash
git clone https://github.com/rystrauss/interact
cd interact
pip install .
```

If you want to use Gym environments that aren't installed by default with Gym, you'll
need to install those yourself (e.g. `pip install gym[atari]`).

## Usage

### Training

An agent can be trained with the following command:

```
python -m interact.train --config <path_to_config_file>
```

This package uses [Gin](https://github.com/google/gin-config) to configure experiments,
and the `--config` option should be a path to a Gin config file. Algorithm-specific
arguments can be found in each agent's documentation.

Some example configuration files can be found in the [`examples`](examples) directory.

### Visualizing Agents

Once an agent has been trained, it can be visualized in its environment with the
following command:

```
python -m interact.play --agent_dir <path/to/agent/dir>
```

where `<path/to/agent/dir>` is the path to the directory that contains the agent you
want to visualize (this is the directory that was created by the training script).

## Implemented Algorithms

* [Advantage Actor-Critic](interact/agents/a2c)
* [Deep Q-Networks](interact/agents/dqn)
* [Proximal Policy Optimization](interact/agents/ppo)
* [Phasic Policy Gradients](interact/agents/ppg)
* [Soft Actor-Critic](interact/agents/sac)
* [Deep Deterministic Policy Gradient](interact/agents/ddpg)
* [Twin Delayed Deep Deterministic Policy Gradient](interact/agents/ddpg)