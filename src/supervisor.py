"""
Supervisor agent for Dreamdeck, orchestrating persona agents and tools
using the langgraph-supervisor pattern.
"""

from src.models import ChatState
from src.agents.registry import AGENT_REGISTRY, get_agent
from src.agents.writer_agent import writer_agent
from langchain_core.messages import HumanMessage, AIMessage
import logging
import sys
import os

# Monkeypatch: Use a no-op task decorator in test environments to avoid langgraph context errors
def _noop_decorator(func):
    return func

import os
if (
    "pytest" in sys.modules
    or "PYTEST_CURRENT_TEST" in os.environ
    or "PYTEST_RUNNING" in os.environ
):
    task = _noop_decorator
else:
    from langgraph.func import task

cl_logger = logging.getLogger("chainlit")

def _normalize_persona(persona: str) -> str:
    if not persona:
        return "default"
    return persona.lower().replace(" ", "_")

async def supervisor(state: ChatState, **kwargs):
    """
    Supervisor agent: routes user input to the correct tool or persona agent.
    Uses the central agent registry for extensibility.
    """
    last_human = state.get_last_human_message()
    if not last_human:
        cl_logger.warning("Supervisor: No user message found in state.")
        return []

    user_input = last_human.content.strip().lower()

    # Tool routing: check for explicit tool commands
    for tool_name in AGENT_REGISTRY:
        if user_input.startswith(f"/{tool_name}") or tool_name in user_input:
            cl_logger.info(f"Supervisor: Routing to {tool_name}_agent.")
            agent = get_agent(tool_name)
            if agent is None:
                cl_logger.warning(f"Supervisor: No agent found for tool '{tool_name}'.")
                continue
            if tool_name == "storyboard":
                # Find last GM message id for storyboard
                gm_msg = next(
                    (msg for msg in reversed(state.messages)
                     if isinstance(msg, AIMessage) and msg.name and "game master" in msg.name.lower()
                     and msg.metadata and "message_id" in msg.metadata),
                    None,
                )
                if gm_msg:
                    return await agent(state, gm_message_id=gm_msg.metadata["message_id"])
                else:
                    cl_logger.warning("Supervisor: No GM message found for storyboard.")
                    return []
            return await agent(state)

    # Default: route to current persona agent
    persona = getattr(state, "current_persona", "default")
    persona_key = _normalize_persona(persona)
    agent = getattr(writer_agent, "persona_agent_registry", {}).get(persona_key, writer_agent)
    cl_logger.info(f"Supervisor: Routing to persona agent '{persona_key}'.")
    return await agent(state)
