# Interact

Interact is a collection of reinforcement learning algorithms.

This package is largely based on and adapted from [OpenAI Baselines](https://github.com/openai/baselines) but is
pared down and written in TensorFlow 2. It is intended to be more intelligible while retaining a certain level of
quality and performance. The hope is that the core pieces of the algorithms are written in such a way so that the
algorithms are easy to parse and conducive to being better understood. The aspects of this package that are most
directly copied from Baselines are related to training infrastructure and efficiency (such as the parallelization
of experience collection) whereas the agents themselves differ the most (and are therefore hopefully more readable).

Reinforcement learning implementations, even the most popular and vetted ones, are known to be plagued by subtle bugs
which do not necessarily make themselves evident. This is almost certainly true of this package, which was put together
by a singular PhD student in some free time, so all of the code should be interpreted with a grain of salt.

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
and passing the `--help` flag with display the available commands.

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