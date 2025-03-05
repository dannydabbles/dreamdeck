import os
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langgraph.message import ToolMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..config import STORYBOARD_EDITOR_AGENT_TEMPERATURE, STORYBOARD_EDITOR_AGENT_MAX_TOKENS, STORYBOARD_EDITOR_AGENT_STREAMING, STORYBOARD_EDITOR_AGENT_VERBOSE, LLM_TIMEOUT

# Initialize logging
cl_logger = logging.getLogger("chainlit")

@task
async def generate_storyboard(content: str) -> str:
    """Generate a storyboard based on the input content.

    Args:
        content (str): The input content for the storyboard.

    Returns:
        str: The generated storyboard.
    """
    try:
        # Placeholder for storyboard generation logic
        return content
    except Exception as e:
        cl_logger.error(f"Storyboard generation failed: {e}")
        return "Error generating storyboard."

# Initialize the storyboard editor agent
storyboard_editor_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=STORYBOARD_EDITOR_AGENT_TEMPERATURE,
        max_tokens=STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
        streaming=STORYBOARD_EDITOR_AGENT_STREAMING,
        verbose=STORYBOARD_EDITOR_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 2,
    ),
    tools=[generate_storyboard],
    checkpointer=MemorySaver(),
)
