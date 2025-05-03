"""
Dreamdeck Supervisor: Leverage langgraph-supervisor for agent orchestration.

This version uses langgraph-supervisor's built-in routing, handoff, and message management.
"""

import logging
import asyncio
import contextvars  # <-- Add this import

import chainlit as cl

import langgraph.config

def _safe_get_config():
    try:
        return langgraph.config.get_config()
    except RuntimeError:
        return {}

from src.agents.decision_agent import decision_agent
from src.agents.dice_agent import dice_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.persona_classifier_agent import persona_classifier_agent
from src.agents.registry import AGENT_REGISTRY, get_agent
from src.agents.report_agent import report_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.todo_agent import todo_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.writer_agent import writer_agent
from src.models import ChatState
from langchain_core.messages import AIMessage  # <-- Add this import

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


# Expose _decide_next_agent for patching in tests
from src.agents.decision_agent import _decide_next_agent

async def supervisor(state: ChatState, **kwargs):
    """
    Entrypoint for the Dreamdeck supervisor.
    Uses the decision agent (oracle) to decide which agent/tool/persona to call next,
    then calls that agent, and (if it's a tool) continues routing until a persona agent is called last.

    This version supports multi-hop routing: it will keep calling the decision agent and routing to tools/personas
    until a persona agent is called last (i.e., the decision agent returns a persona route).
    """
    from src.agents.registry import get_agent

    max_hops = 5  # Prevent infinite loops
    hops = 0
    results = []
    # Clear tool results from previous turns at the start of a new supervisor invocation
    state.tool_results_this_turn = []
    last_route = None
    persona_called = False  # Track if a persona agent has been called in this turn

    while hops < max_hops:
        # Pass current turn's tool results to the decision agent
        decision = await _decide_next_agent(
            state, tool_results_this_turn=state.tool_results_this_turn
        )
        route = decision.get("route", "writer")
        cl_logger.info(f"Supervisor: decision agent routed to '{route}'")
        state.last_agent_called = route  # Track the agent being called

        # --- Check for END route ---
        if route == "END":
            cl_logger.info("Supervisor: END route received, terminating turn.")
            break

        last_route = route

        # --- MULTI-INTENT PATCH: handle list of routes ---
        routes = route if isinstance(route, list) else [route]
        cl_logger.debug(f"Processing routes: {routes}")
        from src.agents.writer_agent import writer_agent

        for single_route in routes:
            cl_logger.debug(f"Processing route: {single_route}")
            if isinstance(single_route, str) and single_route.lower().startswith("persona:"):
                if persona_called:
                    cl_logger.warning("Supervisor: Multiple persona agents in one turn are not allowed. Skipping additional persona agent.")
                    continue  # Skip additional persona agents
                persona_called = True
                persona_name = single_route.split(":", 1)[1].strip() or "Default"
                cl_logger.info(
                    f"Supervisor: switching current_persona to '{persona_name}' based on oracle decision"
                )
                state = state.model_copy(update={"current_persona": persona_name})
                cl.user_session.set("current_persona", persona_name)
                agent_to_call = None
                if (
                    hasattr(writer_agent, "persona_agent_registry")
                    and persona_name in writer_agent.persona_agent_registry
                ):
                    agent_to_call = writer_agent.persona_agent_registry[persona_name]
                else:
                    cl_logger.warning(
                        f"Supervisor: Persona '{persona_name}' not found in writer_agent registry, falling back to default writer."
                    )
                    agent_to_call = writer_agent
                cl_logger.info(f"Persona agent to call: {agent_to_call}")
                if agent_to_call is None:
                    cl_logger.error(
                        f"Supervisor: Could not find agent for persona '{persona_name}'"
                    )
                    continue
                cl_logger.info(f"Calling persona agent: {agent_to_call}")
                # If the agent is a mock (for testing), call it with await
                import unittest.mock
                # --- PATCH: Always await agent_to_call, even if it's a mock ---
                persona_result = await agent_to_call(state)
                if persona_result:
                    results.extend(persona_result)
                    # Immediately add to state messages for storyboard detection
                    state.messages.extend(persona_result)
                    state.tool_results_this_turn.extend(persona_result)
                # Do NOT break here; allow tools after persona agent
            else:
                # Route to the correct tool agent, always using the helper if available
                agent = get_agent(single_route, helper=True)
                if agent is None:
                    cl_logger.warning(
                        f"Supervisor: unknown route '{single_route}', defaulting to writer agent"
                    )
                    from src.agents.writer_agent import writer_agent_helper

                    agent = writer_agent_helper

                # Always call the agent's "_helper" function if it exists, else the agent itself
                agent_helper = None
                if hasattr(agent, "_manage_todo"):
                    agent_helper = getattr(agent, "_manage_todo", None)
                elif hasattr(agent, "_web_search"):
                    agent_helper = getattr(agent, "_web_search", None)
                elif hasattr(agent, "__wrapped__"):
                    agent_helper = getattr(agent, "__wrapped__", None)
                else:
                    if hasattr(agent, "__module__"):
                        try:
                            import importlib

                            agent_mod = importlib.import_module(agent.__module__)
                            helper_name = getattr(agent, "__name__", None)
                            if helper_name and helper_name.endswith("_agent"):
                                helper_func_name = helper_name + "_helper"
                                agent_helper = getattr(agent_mod, helper_func_name, None)
                            if not agent_helper and hasattr(agent_mod, "dice_agent_helper"):
                                agent_helper = getattr(agent_mod, "dice_agent_helper", None)
                            if not agent_helper and hasattr(
                                agent_mod, "web_search_agent_helper"
                            ):
                                agent_helper = getattr(
                                    agent_mod, "web_search_agent_helper", None
                                )
                            if not agent_helper and hasattr(agent_mod, "todo_agent_helper"):
                                agent_helper = getattr(agent_mod, "todo_agent_helper", None)
                            if not agent_helper and hasattr(agent_mod, "writer_agent_helper"):
                                agent_helper = getattr(agent_mod, "writer_agent_helper", None)
                            if not agent_helper and hasattr(
                                agent_mod, "storyboard_editor_agent_helper"
                            ):
                                agent_helper = getattr(
                                    agent_mod, "storyboard_editor_agent_helper", None
                                )
                            if not agent_helper and hasattr(
                                agent_mod, "knowledge_agent_helper"
                            ):
                                agent_helper = getattr(
                                    agent_mod, "knowledge_agent_helper", None
                                )
                            if not agent_helper and hasattr(agent_mod, "report_agent_helper"):
                                agent_helper = getattr(agent_mod, "report_agent_helper", None)
                        except Exception:
                            agent_helper = None

                agent_to_call = agent_helper if agent_helper else agent

                # If the agent is a mock (for testing), call it with await
                import unittest.mock
                # --- PATCH: Always await agent_to_call, even if it's a mock ---
                tool_result = await agent_to_call(state)
                if tool_result:
                    results.extend(tool_result)
                    state.tool_results_this_turn.extend(tool_result)
                    state.messages.extend(tool_result)
        # After processing all routes in this turn, if a persona agent was called, end the turn.
        if persona_called:
            # --- GM Persona Aliases Check ---
            gm_persona_aliases = ["storyteller gm", "game master", "dungeon master"]
            cleaned_persona = state.current_persona.lower().strip()
            cl_logger.debug(f"GM Persona Check: {cleaned_persona} in {gm_persona_aliases}?")
            if cleaned_persona in gm_persona_aliases:
                cl_logger.info(f"Identified as GM persona: {state.current_persona}")
                # Debug: log last 3 message metadatas for troubleshooting
                cl_logger.debug(f"Last 3 messages: {[getattr(msg, 'metadata', None) for msg in state.messages[-3:]]}")
                # Find the GM message *just generated* in the updated state.messages
                gm_message_to_storyboard = None
                # --- CHANGE START ---
                # Search backwards through the updated state messages
                for msg in reversed(state.messages):
                    if isinstance(msg, AIMessage) and msg.metadata and msg.metadata.get("type") == "gm_message":
                        # Check if this message was likely added in the current turn
                        # (A simple check: was it one of the last few messages?)
                        # More robust checks could involve timestamps or turn counters if needed.
                        # For now, assume the last matching message is the one.
                        gm_message_to_storyboard = msg
                        break # Found the most recent suitable message
                # --- CHANGE END ---

                if gm_message_to_storyboard and gm_message_to_storyboard.metadata.get("message_id"):
                    gm_message_id = gm_message_to_storyboard.metadata["message_id"]
                    cl_logger.info(f"Generating storyboard for message ID: {gm_message_id}")
                    from src.config import STABLE_DIFFUSION_API_URL
                    # Ensure storyboard agent is awaited and track the task
                    # Use contextvars.copy_context() to propagate context into the background task
                    ctx = contextvars.copy_context()
                    storyboard_task = asyncio.create_task(
                        ctx.run(
                            storyboard_editor_agent,
                            state,
                            gm_message_id=gm_message_id,
                            sd_api_url=STABLE_DIFFUSION_API_URL
                        )
                    )
                    # Check if background_tasks exists before appending
                    if hasattr(state, 'background_tasks') and isinstance(state.background_tasks, list):
                        state.background_tasks.append(storyboard_task)
                    else:
                        cl_logger.warning("State object missing 'background_tasks' list attribute.")

                else:
                    cl_logger.warning("No suitable GM message found in the current step's result for storyboard generation")
            else:
                cl_logger.info(f"Current persona '{state.current_persona}' is not a GM persona")
            # Break the loop after a persona agent has been called and processed for the turn
            break

        hops += 1

    if hops >= max_hops:
        cl_logger.warning(
            "Supervisor: max hops reached, ending turn with current results."
        )

    return results


# Patch: add .ainvoke for test compatibility (LangGraph expects this in tests)
supervisor.ainvoke = supervisor

# Expose _decide_next_agent for patching in tests
supervisor._decide_next_agent = _decide_next_agent


def _noop_task(x):
    return x


supervisor.task = _noop_task


def task(x):
    return x

async def _with_safe_config(fn, *args, **kwargs):
    """Run a function with a safe LangGraph config context."""
    import langgraph.config
    original_get_config = langgraph.config.get_config
    langgraph.config.get_config = _safe_get_config
    try:
        return await fn(*args, **kwargs)
    finally:
        langgraph.config.get_config = original_get_config
