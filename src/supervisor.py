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

# Gather all persona and tool agents
if hasattr(writer_agent, "persona_agent_registry"):
    persona_agents = list(writer_agent.persona_agent_registry.values())
else:
    persona_agents = [writer_agent]

tool_agents = [entry["agent"] for tool, entry in AGENT_REGISTRY.items()]

# Use the writer agent's LLM if available, else fallback to gpt-4o
try:
    model = writer_agent.llm
except AttributeError:
    model = ChatOpenAI(model="gpt-4o")

# Use langgraph-supervisor's built-in routing and handoff
supervisor_workflow = create_supervisor(
    persona_agents + tool_agents,
    model=model,
    # Optionally, you can set output_mode, prompt, or custom tools here
    # output_mode="last_message",
).compile()

# Entrypoint for Chainlit and tests
async def supervisor(state: ChatState, **kwargs):
    """
    Entrypoint for the Dreamdeck supervisor using langgraph-supervisor.
    """
    return await supervisor_workflow.ainvoke(state, **kwargs)

# Patch: add .ainvoke for test compatibility (LangGraph expects this in tests)
supervisor.ainvoke = supervisor_workflow.ainvoke
