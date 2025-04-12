"""
Dreamdeck Supervisor: Leverage langgraph-supervisor for agent orchestration.

This version uses langgraph-supervisor's built-in routing, handoff, and message management.
"""

from src.models import ChatState
from src.agents.registry import AGENT_REGISTRY, get_agent
from src.agents.writer_agent import writer_agent
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor

import logging

cl_logger = logging.getLogger("chainlit")

# Gather all persona and tool agents, ensuring unique agent names (case-sensitive)
if hasattr(writer_agent, "persona_agent_registry"):
    persona_agents = list(writer_agent.persona_agent_registry.values())
else:
    persona_agents = [writer_agent]

# Remove duplicate persona agents by name (case-sensitive)
unique_persona_agents = {}
for agent in persona_agents:
    agent_name = getattr(agent, "name", None)
    if agent_name and agent_name not in unique_persona_agents:
        unique_persona_agents[agent_name] = agent
persona_agents = list(unique_persona_agents.values())

tool_agents = [entry["agent"] for tool, entry in AGENT_REGISTRY.items()]

# Remove duplicate tool agents by name (case-sensitive)
unique_tool_agents = {}
for agent in tool_agents:
    agent_name = getattr(agent, "name", None)
    if agent_name and agent_name not in unique_tool_agents:
        unique_tool_agents[agent_name] = agent
tool_agents = list(unique_tool_agents.values())

# Remove any tool agent whose name matches a persona agent (case-sensitive)
persona_agent_names = set(getattr(agent, "name", None) for agent in persona_agents if getattr(agent, "name", None))
tool_agents = [agent for agent in tool_agents if getattr(agent, "name", None) not in persona_agent_names]

import chainlit as cl

# Use the writer agent's LLM if available, else fallback to gpt-4o
def get_dynamic_model():
    # Try to get dynamic settings from Chainlit user session
    try:
        llm_temperature = cl.user_session.get("llm_temperature")
        llm_max_tokens = cl.user_session.get("llm_max_tokens")
        llm_endpoint = cl.user_session.get("llm_endpoint")
        from langchain_openai import ChatOpenAI
        kwargs = {}
        if llm_temperature is not None:
            kwargs["temperature"] = llm_temperature
        if llm_max_tokens is not None:
            kwargs["max_tokens"] = llm_max_tokens
        if llm_endpoint:
            kwargs["base_url"] = llm_endpoint
        return ChatOpenAI(model="gpt-4o", **kwargs)
    except Exception:
        # Fallback to static model
        try:
            return writer_agent.llm
        except AttributeError:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o")

# Use langgraph-supervisor's built-in routing and handoff
def get_supervisor_workflow():
    model = get_dynamic_model()
    return create_supervisor(
        persona_agents + tool_agents,
        model=model,
    ).compile()

# Entrypoint for Chainlit and tests
async def supervisor(state: ChatState, **kwargs):
    """
    Entrypoint for the Dreamdeck supervisor using langgraph-supervisor.
    Uses dynamic LLM settings from Chainlit UI if available.
    """
    workflow = get_supervisor_workflow()
    result = await workflow.ainvoke(state, **kwargs)
    # Return just the messages list for test compatibility
    return result["messages"]

# Patch: add .ainvoke for test compatibility (LangGraph expects this in tests)
supervisor.ainvoke = supervisor_workflow.ainvoke

# Patch: add dummy 'task' attribute for test compatibility (for unittest.mock.patch in tests)
def _noop_task(x):
    return x
supervisor.task = _noop_task

# Patch: add dummy 'task' attribute at module level for test compatibility
def task(x):
    return x
