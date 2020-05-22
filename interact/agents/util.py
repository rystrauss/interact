"""Utilities for working with agents.

Author: Ryan Strauss
"""

from interact.agents.a2c.a2c import A2CAgent

_AGENTS = {
    'a2c': A2CAgent
}


def available_agents():
    """Returns a list of all available agent names.

    Returns:
        A list of strings indicating the agents that are available.
    """
    return list(_AGENTS.keys())


def get_agent(name):
    """Gets the corresponding agent class for a given name.

    Args:
        name: The name of the agent to be fetched.

    Returns:
        The agent class that corresponds to the given name.
    """
    if name not in _AGENTS:
        raise ValueError(f'{name} is not a valid agent -- choose from {list(_AGENTS.keys())}')

    return _AGENTS[name]
