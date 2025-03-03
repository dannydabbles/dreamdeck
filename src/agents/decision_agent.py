import os
import logging
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from agents.dice_agent import dice_roll_agent
from .agents.web_search_agent import web_search_agent
from .config import (
    DECISION_AGENT_TEMPERATURE,
    DECISION_AGENT_MAX_TOKENS,
    DECISION_AGENT_STREAMING,
    DECISION_AGENT_VERBOSE,
    LLM_TIMEOUT,
)
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver

# Initialize logging
cl_logger = logging.getLogger("chainlit")

def decide_action(user_input: str) -> ToolMessage:
    """Determine the next action based on user input.

    Args:
        user_input (str): The user's input.

    Returns:
        ToolMessage: The next action to take.
    """
    try:
        # Determine the action based on user input
        if "roll" in user_input.lower():
            return ToolMessage(content="roll")
        elif "search" in user_input.lower():
            return ToolMessage(content="search")
        else:
            return ToolMessage(content="continue_story")
    except Exception as e:
        cl_logger.error(f"Decision failed: {e}", exc_info=True)
        return ToolMessage(content="continue_story")

# Initialize the decision agent
decision_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=DECISION_AGENT_TEMPERATURE,
        max_tokens=DECISION_AGENT_MAX_TOKENS,
        streaming=DECISION_AGENT_STREAMING,
        verbose=DECISION_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 2,
    ),
    tools=[decide_action, dice_roll_agent, web_search_agent],
    checkpointer=MemorySaver(),
)
