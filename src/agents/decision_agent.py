from src.config import config
import os
import logging
from langgraph.func import task
from langgraph.prebuilt import create_react_agent
from .dice_agent import dice_roll  # Import the tool, not the agent
from .web_search_agent import web_search  # Import the tool, not the agent
from ..config import (
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

async def _decide_action(user_input: str, store=None, previous=None) -> dict:
    """Determine the next action based on user input.

    Args:
        user_input (str): The user's input.
        store (BaseStore, optional): The store for chat state. Defaults to None.
        previous (ChatState, optional): Previous chat state. Defaults to None.

    Returns:
        dict: The next action to take.
    """
    try:
        if "roll" in user_input.lower():
            return {"name": "roll", "args": {}}
        elif "search" in user_input.lower():
            return {"name": "search", "args": {}}
        else:
            return {"name": "continue_story", "args": {}}
    except Exception as e:
        cl_logger.error(f"Decision failed: {e}")
        return {"name": "continue_story", "args": {}}

@task
async def decide_action(user_input: str, **kwargs) -> dict:
    return await _decide_action(user_input)

# Expose the function as decision_agent
decision_agent = decide_action
