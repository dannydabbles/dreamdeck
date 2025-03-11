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
    DECISION_PROMPT,
)
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from langchain_core.messages import HumanMessage  # Import HumanMessage

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _decide_action(user_message: HumanMessage) -> dict:
    """Determine the next action based on user input.

    Args:
        user_message (HumanMessage): The user's input.

    Returns:
        dict: The next action to take.
    """
    try:
        formatted_prompt = DECISION_PROMPT.format(
            user_input=user_message.content
        )

        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=DECISION_AGENT_TEMPERATURE,
            max_tokens=DECISION_AGENT_MAX_TOKENS,
            streaming=DECISION_AGENT_STREAMING,
            verbose=DECISION_AGENT_VERBOSE,
            timeout=LLM_TIMEOUT
        )

        # Generate the decision
        response = await llm.agenerate([formatted_prompt])
        decision = response.generations[0][0].text.strip()

        if "roll" in decision.lower():
            return {"name": "roll", "args": {}}
        elif "search" in decision.lower():
            return {"name": "search", "args": {}}
        else:
            return {"name": "continue_story", "args": {}}
    except Exception as e:
        cl_logger.error(f"Decision failed: {e}")
        return {"name": "continue_story", "args": {}}

@task
async def decide_action(user_message: HumanMessage, **kwargs) -> dict:
    return await _decide_action(user_message)

# Expose the function as decision_agent
decision_agent = decide_action
