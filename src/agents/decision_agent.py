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
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # Import HumanMessage
from ..models import ChatState


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
    result = []

    # Get last human message
    user_input = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    try:
        formatted_prompt = DECISION_PROMPT.format(
            user_input=user_input.content
        )

        # Initialize the LLM
        llm = ChatOpenAI(
            base_url="http://192.168.1.111:5000/v1",
            temperature=DECISION_AGENT_TEMPERATURE,
            max_tokens=DECISION_AGENT_MAX_TOKENS,
            streaming=DECISION_AGENT_STREAMING,
            verbose=DECISION_AGENT_VERBOSE,
            timeout=LLM_TIMEOUT
        )

        # Generate the decision
        response = llm.invoke([("system", formatted_prompt)])
        cl_logger.info(f"Decision response: {response.content}")
        decision = response.content.strip()

        if "roll" in decision.lower():
            result = AIMessage(content="Rolling dice...", name="dice_roll")
        elif "search" in decision.lower():
            result = AIMessage(content="Searching the web...", name="web_search")
        else:
            result = AIMessage(content="Continuing the story...", name="continue_story")
    except Exception as e:
        cl_logger.error(f"Decision failed: {e}")
        result = AIMessage(content="Decision failed.", name="error")

    return [result]

@task
async def decide_action(state: ChatState, **kwargs) -> list[BaseMessage]:
    return await _decide_action(state)

# Expose the function as decision_agent
decision_agent = decide_action
