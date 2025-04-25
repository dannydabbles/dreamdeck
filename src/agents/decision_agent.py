import logging

from jinja2 import Template

from src.config import LLM_TIMEOUT, OPENAI_SETTINGS, config

# --- PATCH: Monkeypatch langgraph.config.get_config to avoid "outside of a runnable context" error ---
try:
    import langgraph.config

    def _safe_get_config():
        try:
            return langgraph.config.get_config()
        except Exception:
            return {}

    langgraph.config.get_config = _safe_get_config
except ImportError:
    pass

import chainlit as cl
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.func import task

from src.models import ChatState

cl_logger = logging.getLogger("chainlit")


@cl.step(name="Decision Agent", type="tool")
async def _decide_next_agent(
    state: ChatState, tool_results_this_turn: list = None, **kwargs
) -> dict:
    """
    Uses an LLM to decide which agent/persona/tool should handle the next turn.
    Returns a dict: {"route": "dice"|"search"|"writer"|"persona:Therapist"|...}
    """
    try:
        template = Template(config.loaded_prompts["oracle_decision_prompt"])
        # Prepare context string for tool results this turn
        tool_results_this_turn_str = ""
        if tool_results_this_turn:
            tool_results_this_turn_str = "\n".join(
                [f"{msg.name}: {msg.content}" for msg in tool_results_this_turn]
            )
        prompt = template.render(
            recent_chat_history=state.get_recent_history_str(),
            tool_results_this_turn=tool_results_this_turn_str,
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
            user_preferences=state.user_preferences,
        )

        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("decision_temp", 0.2)
        final_endpoint = user_settings.get("decision_endpoint") or OPENAI_SETTINGS.get(
            "base_url"
        )
        final_max_tokens = user_settings.get("decision_max_tokens", 100)

        llm = ChatOpenAI(
            model=config.llm.model,
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=False,
            verbose=True,
            timeout=LLM_TIMEOUT,
        )

        response = await llm.ainvoke([("system", prompt)])
        content = response.content.strip()

        # Remove markdown code fencing if present
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()

        import json

        route = None
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                route = parsed.get("route")
        except Exception:
            # fallback: treat as string
            route = content

        # --- MULTI-INTENT PATCH ---
        # If the route is a comma-separated string, split into a list
        if isinstance(route, str) and "," in route:
            route = [r.strip() for r in route.split(",") if r.strip()]
        # If the route is a string containing " and ", split as well
        elif isinstance(route, str) and " and " in route:
            route = [r.strip() for r in route.split(" and ") if r.strip()]

        # --- FALLBACK: Add web_search if user intent is obvious ---
        last_human = state.get_last_human_message()
        if last_human:
            user_text = last_human.content.lower()
            search_keywords = ["look up", "search", "research"]
            if any(kw in user_text for kw in search_keywords):
                # If route is a string, make it a list
                if isinstance(route, str):
                    route = [route]
                # If web_search/search not already in the route, add it
                if not any(r in ["web_search", "search"] for r in route):
                    route = ["web_search"] + route

        if not route:
            route = "writer"

        cl_logger.info(f"Decision agent suggests route: {route}")
        return {"route": route}

    except Exception as e:
        cl_logger.error(f"Decision agent failed: {e}")
        return {"route": "writer"}


# Refactored: decision_agent is now a stateless, LLM-backed function (task)
@task
async def decision_agent(state: ChatState, **kwargs) -> dict:
    # Note: tool_results_this_turn is passed by the supervisor directly
    return await _decide_next_agent(
        state, tool_results_this_turn=state.tool_results_this_turn, **kwargs
    )
