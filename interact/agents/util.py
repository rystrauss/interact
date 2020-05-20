from interact.agents.a2c.a2c import A2CAgent

_AGENTS = {
    'a2c': A2CAgent
}


def available_agents():
    return list(_AGENTS.keys())


def get_agent(name):
    if name not in _AGENTS:
        raise ValueError(f'{name} is not a valid agent -- choose from {list(_AGENTS.keys())}')

    return _AGENTS[name]
