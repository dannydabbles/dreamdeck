from src.config import config
import json
import logging
from jinja2 import Template
from langgraph.func import task
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)
from src.config import (
    DECISION_AGENT_TEMPERATURE,
    DECISION_AGENT_MAX_TOKENS,
    DECISION_AGENT_STREAMING,
    DECISION_AGENT_VERBOSE,
    LLM_TIMEOUT,
    DECISION_AGENT_BASE_URL,
)
import chainlit as cl

cl_logger = logging.getLogger("chainlit")


async def _decide_actions(state) -> list[str]:
    """
    Determine the ordered list of actions to perform based on user input.

    Returns a list of action names, e.g., ["search", "roll", "write"].
    """
    messages = state.messages

    user_input = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )

    if not user_input:
        return ["continue_story"]

    try:
        # Compose prompt with examples
        prompt_template = """
You are an AI orchestrator that decides which tools to invoke based on the user's latest input and chat history.

Output a JSON object with an ordered list of actions to perform, e.g.:

{"actions": ["search", "roll", "write"]}

Possible actions include:
- "search"
- "roll"
- "todo"
- "character"
- "lore"
- "puzzle"
- "write"
- "continue_story" (default if no tools needed)

Always output a JSON object with an "actions" list, ordered by execution priority.

If no tools are needed, output:

{"actions": ["continue_story"]}

User input:
\"\"\"{user_input}\"\"\"

Chat history:
\"\"\"{chat_history}\"\"\"
"""
        formatted_prompt = prompt_template.format(
            user_input=user_input.content,
            chat_history=state.get_recent_history_str(),
        )

        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("decision_temp", DECISION_AGENT_TEMPERATURE)
        final_endpoint = user_settings.get("decision_endpoint") or DECISION_AGENT_BASE_URL
        final_max_tokens = user_settings.get("decision_max_tokens", DECISION_AGENT_MAX_TOKENS)

        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=DECISION_AGENT_STREAMING,
            verbose=DECISION_AGENT_VERBOSE,
            timeout=LLM_TIMEOUT,
        )

        response = await llm.ainvoke([("system", formatted_prompt)])
        cl_logger.info(f"Orchestrator response: {response.content}")

        try:
            parsed = json.loads(response.content.strip())
            actions = parsed.get("actions", [])
            if not isinstance(actions, list) or not actions:
                return ["continue_story"]
            return actions
        except json.JSONDecodeError:
            cl_logger.warning("Failed to parse orchestrator JSON, defaulting to continue_story")
            return ["continue_story"]

    except Exception as e:
        cl_logger.error(f"Orchestrator failed: {e}")
        return ["continue_story"]


@task
async def orchestrate(state, **kwargs) -> list[str]:
    return await _decide_actions(state)


orchestrator_agent = orchestrate
