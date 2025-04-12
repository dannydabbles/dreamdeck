"""
Dreamdeck Supervisor: Leverage langgraph-supervisor for agent orchestration.

This version uses langgraph-supervisor's built-in routing, handoff, and message management.
"""

from src.models import ChatState
from src.agents.registry import AGENT_REGISTRY, get_agent
from src.agents.writer_agent import writer_agent
from src.agents.dice_agent import dice_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.todo_agent import todo_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.report_agent import report_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.persona_classifier_agent import persona_classifier_agent
from src.agents.decision_agent import decision_agent

import logging
import chainlit as cl

cl_logger = logging.getLogger("chainlit")

# Map tool/agent names to callables for routing
AGENT_MAP = {
    "dice": dice_agent,
    "roll": dice_agent,
    "web_search": web_search_agent,
    "search": web_search_agent,
    "todo": todo_agent,
    "knowledge": knowledge_agent,
    "report": report_agent,
    "storyboard": storyboard_editor_agent,
    "persona_classifier": persona_classifier_agent,
    "decision": decision_agent,
    "writer": writer_agent,
    "storyteller_gm": writer_agent,
    "therapist": get_agent("therapist"),
    "secretary": get_agent("secretary"),
    "coder": get_agent("coder"),
    "friend": get_agent("friend"),
    "lorekeeper": get_agent("lorekeeper"),
    "dungeon_master": get_agent("dungeon_master"),
    "default": writer_agent,
}

# Use the writer agent's LLM if available, else fallback to gpt-4o
def get_dynamic_model():
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
        try:
            return writer_agent.llm
        except AttributeError:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o")

async def supervisor(state: ChatState, **kwargs):
    """
    Entrypoint for the Dreamdeck supervisor.
    Uses the decision agent (oracle) to decide which agent/tool/persona to call next,
    then calls that agent, and (if it's a tool) follows up with the GM persona if appropriate.

    This version avoids langgraph context issues by always calling the agent's *_helper function
    (if available) instead of the @task-decorated function.

    Additionally, the supervisor/oracle will also decide and set the current persona based on the conversation.
    """
    # 1. Use the decision agent to decide what to do next
    # Always use the helper to avoid langgraph context issues
    from src.agents.decision_agent import _decide_next_agent
    decision = await _decide_next_agent(state)
    route = decision.get("route", "writer")
    cl_logger.info(f"Supervisor: decision agent routed to '{route}'")

    # 1b. If the route is a persona (e.g., "persona:Therapist"), update state.current_persona accordingly
    if route.startswith("persona:"):
        persona_name = route.split(":", 1)[1]
        cl_logger.info(f"Supervisor: switching current_persona to '{persona_name}' based on oracle decision")
        state.current_persona = persona_name
        import chainlit as cl
        cl.user_session.set("current_persona", persona_name)

    # 2. Route to the correct agent/tool/persona, always using the helper if available
    from src.agents.registry import get_agent
    agent = get_agent(route, helper=True)
    if agent is None:
        cl_logger.warning(f"Supervisor: unknown route '{route}', defaulting to writer agent")
        from src.agents.writer_agent import writer_agent_helper
        agent = writer_agent_helper

    # 3. Always call the agent's "_helper" function if it exists, else the agent itself
    # This avoids LangGraph context errors from @task-decorated functions
    agent_helper = None
    if hasattr(agent, "__module__"):
        try:
            import importlib
            agent_mod = importlib.import_module(agent.__module__)
            helper_name = getattr(agent, "__name__", None)
            if helper_name and helper_name.endswith("_agent"):
                helper_func_name = helper_name + "_helper"
                agent_helper = getattr(agent_mod, helper_func_name, None)
            # Fallback: try common helper names
            if not agent_helper and hasattr(agent_mod, "dice_agent_helper"):
                agent_helper = getattr(agent_mod, "dice_agent_helper", None)
            if not agent_helper and hasattr(agent_mod, "web_search_agent_helper"):
                agent_helper = getattr(agent_mod, "web_search_agent_helper", None)
            if not agent_helper and hasattr(agent_mod, "todo_agent_helper"):
                agent_helper = getattr(agent_mod, "todo_agent_helper", None)
            if not agent_helper and hasattr(agent_mod, "writer_agent_helper"):
                agent_helper = getattr(agent_mod, "writer_agent_helper", None)
            if not agent_helper and hasattr(agent_mod, "storyboard_editor_agent_helper"):
                agent_helper = getattr(agent_mod, "storyboard_editor_agent_helper", None)
            if not agent_helper and hasattr(agent_mod, "knowledge_agent_helper"):
                agent_helper = getattr(agent_mod, "knowledge_agent_helper", None)
            if not agent_helper and hasattr(agent_mod, "report_agent_helper"):
                agent_helper = getattr(agent_mod, "report_agent_helper", None)
        except Exception:
            agent_helper = None

    # If a helper function exists, use it; else use the agent directly
    agent_to_call = agent_helper if agent_helper else agent

    if route in ("dice", "roll", "web_search", "search", "todo", "knowledge", "report", "storyboard"):
        # Tool agent: call tool, then follow up with GM if appropriate
        tool_result = await agent_to_call(state)
        # If the tool result is not an error, follow up with the GM persona
        if tool_result and getattr(tool_result[0], "name", "") != "error":
            # Optionally, update state with tool result before calling GM
            state.messages.extend(tool_result)
            # Call the GM persona (writer agent) to narrate or react
            from src.agents.writer_agent import writer_agent_helper
            gm_result = await writer_agent_helper(state)
            return tool_result + gm_result
        else:
            return tool_result
    else:
        # Persona agent: just call it
        return await agent_to_call(state)

# Patch: add .ainvoke for test compatibility (LangGraph expects this in tests)
supervisor.ainvoke = supervisor

def _noop_task(x):
    return x
supervisor.task = _noop_task

def task(x):
    return x
