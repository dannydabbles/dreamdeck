import pytest

from src.agents.registry import AGENT_REGISTRY, get_agent, list_agents


def test_get_agent_known():
    for name in AGENT_REGISTRY:
        agent = get_agent(name)
        assert agent is not None, f"Agent {name} should be found"


def test_get_agent_unknown():
    assert get_agent("notarealagent") is None


def test_list_agents():
    agents = list_agents()
    assert isinstance(agents, list)
    assert all(isinstance(t, tuple) and len(t) == 2 for t in agents)
    names = [name for name, _ in agents]
    for name in AGENT_REGISTRY:
        assert name in names
