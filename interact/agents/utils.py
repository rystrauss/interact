from typing import List, Type

from interact.agents.base import Agent

_mapping = {}


def register(name: str):
    """Decorator that registers an agent so it can be accessed through the command line interface."""

    def _thunk(cls):
        _mapping[name] = cls
        return cls

    return _thunk


def available_agents() -> List[str]:
    """Returns a list of all available agent names.

    Returns:
        A list of strings indicating the agents that are available.
    """
    return list(_mapping.keys())


def get_agent(name: str) -> Type[Agent]:
    """Gets the corresponding agent class for a given name.

    Args:
        name: The name of the agent to be fetched.

    Returns:
        The agent class that corresponds to the given name.
    """
    if name not in _mapping:
        raise ValueError(
            f"{name} is not a valid agent -- choose from {available_agents()}"
        )

    return _mapping[name]
