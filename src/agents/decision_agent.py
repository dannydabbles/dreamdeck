from src.config import config
import os
import logging
from jinja2 import Template
from langgraph.func import task
from langgraph.prebuilt import create_react_agent
from .dice_agent import dice_roll  # Import the tool, not the agent
from .web_search_agent import web_search  # Import the tool, not the agent
from src.config import (
    DECISION_AGENT_TEMPERATURE,
    DECISION_AGENT_MAX_TOKENS,
    DECISION_AGENT_STREAMING,
    DECISION_AGENT_VERBOSE,
    LLM_TIMEOUT,
    DECISION_AGENT_BASE_URL,
    DECISION_PROMPT,
)
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)  # Import HumanMessage
from src.models import ChatState

import chainlit as cl

# Initialize logging
cl_logger = logging.getLogger("chainlit")


async def _decide_action(state: ChatState) -> list[BaseMessage]:
    """Determine the next action based on user input.

    Args:
        user_message (HumanMessage): The user's input.

    Returns:
        dict: The next action to take.
    """
    messages = state.messages

    # Get last human message
    user_input = next(
        (m for m in reversed(messages) if isinstance(m, HumanMessage)), None
    )

    if not user_input:
        return [AIMessage(content="No user input found.", name="error")]

    try:
        template = Template(DECISION_PROMPT)
        formatted_prompt = template.render(user_input=user_input.content)

        # Get user settings and defaults
        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("decision_temp", DECISION_AGENT_TEMPERATURE)
        final_endpoint = user_settings.get("decision_endpoint") or DECISION_AGENT_BASE_URL
        final_max_tokens = user_settings.get("decision_max_tokens", DECISION_AGENT_MAX_TOKENS)

        # Initialize the LLM with potentially overridden settings
        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=DECISION_AGENT_STREAMING,
            verbose=DECISION_AGENT_VERBOSE,
            timeout=LLM_TIMEOUT,
        )

        # Generate the decision
        response = await llm.ainvoke([("system", formatted_prompt)])
        cl_logger.info(f"Decision response: {response.content}")
        decision = response.content.strip()

        if "roll" in decision.lower():
            result = [AIMessage(content="Rolling dice...", name="dice_roll")]
        elif "search" in decision.lower():
            result = [AIMessage(content="Searching the web...", name="web_search")]
        else:
            result = [AIMessage(content="Continuing the story...", name="continue_story")]
    except Exception as e:
        cl_logger.error(f"Decision failed: {e}")
        result = [AIMessage(content="Decision failed.", name="error")]

    return result


@task
async def decide_action(state: ChatState, **kwargs) -> list[BaseMessage]:
    return await _decide_action(state)


# Expose the function as decision_agent
decision_agent = decide_action
