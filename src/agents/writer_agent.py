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

@task
async def generate_story(content: str) -> str:
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

# Initialize the writer AI agent
writer_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=WRITER_AGENT_TEMPERATURE,
        max_tokens=WRITER_AGENT_MAX_TOKENS,
        streaming=WRITER_AGENT_STREAMING,
        verbose=WRITER_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 3,
    ),
    tools=[generate_story],
    checkpointer=MemorySaver(),
)
