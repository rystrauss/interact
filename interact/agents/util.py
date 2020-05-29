"""Utilities for working with agents.

Author: Ryan Strauss
"""

_mapping = {}


def register(name):
    """Decorator that registers an agent so it can be accessed through the command line interface."""
    def _thunk(cls):
        _mapping[name] = cls
        return cls

    return _thunk


def available_agents():
    """Returns a list of all available agent names.

    Returns:
        A list of strings indicating the agents that are available.
    """
    return list(_mapping.keys())


def get_agent(name):
    """Gets the corresponding agent class for a given name.

    Args:
        name: The name of the agent to be fetched.

    Returns:
        The agent class that corresponds to the given name.
    """
    if name not in _mapping:
        raise ValueError(f'{name} is not a valid agent -- choose from {list(_mapping.keys())}')

    return _mapping[name]
