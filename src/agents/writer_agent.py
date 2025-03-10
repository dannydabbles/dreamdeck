from src.config import config
import os
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..config import WRITER_AGENT_TEMPERATURE, WRITER_AGENT_MAX_TOKENS, WRITER_AGENT_STREAMING, WRITER_AGENT_VERBOSE, LLM_TIMEOUT

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _generate_story(content: str) -> str:
    """Generate a story segment based on the input content.

    Args:
        content (str): The input content for the story.

    Returns:
        str: The generated story segment.
    """
    try:
        # Placeholder for story generation logic
        return content
    except Exception as e:
        cl_logger.error(f"Story generation failed: {e}")
        return "Error generating story."

@task
async def generate_story(content: str, **kwargs) -> str:
    return await _generate_story(content)

writer_agent = generate_story  # Expose the function as writer_agent
