# Interact

Interact contains implementations of several deep reinforcement learning algorithms.

## Installation

Interact can be installed with the steps below.

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

### Training

An agent can be trained with the following command:
```bash
python -m interact.train --config <path_to_config_file>
```

This package uses [Gin](https://github.com/google/gin-config) to configure experiments, and the `--config` option should
be a path to a Gin config file. Algorithm-specific arguments can be found in each agent's documentation.

Some example configuration files can be found in the [`examples`](examples) directory.


## Implemented Algorithms

* [A2C](interact/agents/a2c)