# Interact

Interact is a collection of reinforcement learning algorithms.

This package is largely based on and adapted from [OpenAI Baselines](https://github.com/openai/baselines) but is
pared down and written in TensorFlow 2.

## Installation

Interact can be installed with the steps below. Note that Interact requires Python 3, which should be preferred
over the no longer maintained Python 2. It is also recommended that the package be installed within a virtual
environment.

* Clone the repository.
```bash
git clone https://github.com/rystrauss/interact
cd interact
```

* Install the package.
```bash
pip install -e .
```

## Usage

The Interact command line interface can be accessed with:
```bash
python -m interact.run
```
and passing the `--help` flag will display the available commands.

An agent can be trained as follows:
```bash
python -m interact.run train --agent <name_of_the_algorithm> --env <environment_id> [additional arguments]
```
Algorithm-specific additional arguments can be found in each agent's documentation.

After an agent has been trained, it can be rendered while acting in its environment as follows:
```bash
python -m interact.run play --dir <path_to_the_agents_log_directory>
```

## Implemented Algorithms

* [A2C](interact/agents/a2c)
* [DQN](interact/agents/deepq)
* [PPO](interact/agents/ppo)
* [PPG](interact/agents/ppg)
