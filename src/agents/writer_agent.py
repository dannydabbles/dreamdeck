from src.config import config
import os
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..config import WRITER_AGENT_TEMPERATURE, WRITER_AGENT_MAX_TOKENS, WRITER_AGENT_STREAMING, WRITER_AGENT_VERBOSE, LLM_TIMEOUT, AI_WRITER_PROMPT

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _generate_story(content: str, store=None, previous=None) -> str:
    """Generate a story segment based on the input content.

    Args:
        content (str): The input content for the story.
        store (BaseStore, optional): The store for chat state. Defaults to None.
        previous (ChatState, optional): Previous chat state. Defaults to None.

    Returns:
        str: The generated story segment.
    """
    try:
        formatted_prompt = AI_WRITER_PROMPT.format(
            recent_chat_history=previous.get_recent_history_str(),
            memories=previous.get_memories_str(),
            tool_results=previous.get_tool_results_str()
        )

        # Initialize the LLM
        llm = ChatOpenAI(
            temperature=WRITER_AGENT_TEMPERATURE,
            max_tokens=WRITER_AGENT_MAX_TOKENS,
            streaming=WRITER_AGENT_STREAMING,
            verbose=WRITER_AGENT_VERBOSE,
            timeout=LLM_TIMEOUT
        )

        # Generate the story
        response = await llm.agenerate([formatted_prompt])
        story_segment = response.generations[0][0].text.strip()

        return story_segment
    except Exception as e:
        cl_logger.error(f"Story generation failed: {e}")
        return "Error generating story."

@task
async def generate_story(content: str, **kwargs) -> str:
    return await _generate_story(content)

writer_agent = generate_story  # Expose the function as writer_agent
