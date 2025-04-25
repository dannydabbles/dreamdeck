"""
Dreamdeck Supervisor: Leverage langgraph-supervisor for agent orchestration.

This version uses langgraph-supervisor's built-in routing, handoff, and message management.
"""

import logging

import chainlit as cl

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
    then calls that agent, and (if it's a tool) continues routing until a persona agent is called last.

    This version supports multi-hop routing: it will keep calling the decision agent and routing to tools/personas
    until a persona agent is called last (i.e., the decision agent returns a persona route).
    """
    from src.agents.decision_agent import _decide_next_agent
    from src.agents.registry import get_agent

    max_hops = 5  # Prevent infinite loops
    hops = 0
    results = []
    # Clear tool results from previous turns at the start of a new supervisor invocation
    state.tool_results_this_turn = []
    last_route = None

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
        from src.agents.writer_agent import writer_agent

        for single_route in routes:
            if isinstance(single_route, str) and single_route.lower().startswith("persona:"):
                persona_name = single_route.split(":", 1)[1].strip() or "Default"
                cl_logger.info(
                    f"Supervisor: switching current_persona to '{persona_name}' based on oracle decision"
                )
                state.current_persona = persona_name
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
                if agent_to_call is None:
                    cl_logger.error(
                        f"Supervisor: Could not find agent for persona '{persona_name}'"
                    )
                    break
                persona_result = await agent_to_call(state)
                if persona_result:
                    results.extend(persona_result)
                # After calling a persona agent, always break (persona should be last)
                break
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

                tool_result = await agent_to_call(state)
                if tool_result:
                    results.extend(tool_result)
                    state.tool_results_this_turn.extend(tool_result)
                    state.messages.extend(tool_result)
                # Defensive: If the agent called was the writer agent or any persona agent, break the loop
                if (
                    agent_to_call == writer_agent
                    or getattr(agent_to_call, "__class__", None) == writer_agent.__class__
                    or getattr(agent_to_call, "persona_name", None) in getattr(writer_agent, "persona_agent_registry", {})
                ):
                    cl_logger.info("Supervisor: Writer or persona agent called, ending turn to prevent repeated narrative responses.")
                    break
        hops += 1

    if hops >= max_hops:
        cl_logger.warning(
            "Supervisor: max hops reached, ending turn with current results."
        )

    return results


# Patch: add .ainvoke for test compatibility (LangGraph expects this in tests)
supervisor.ainvoke = supervisor


def _noop_task(x):
    return x


supervisor.task = _noop_task


def task(x):
    return x
