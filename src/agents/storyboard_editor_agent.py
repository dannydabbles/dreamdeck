from src.config import config
import os
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..config import STORYBOARD_EDITOR_AGENT_TEMPERATURE, STORYBOARD_EDITOR_AGENT_MAX_TOKENS, STORYBOARD_EDITOR_AGENT_STREAMING, STORYBOARD_EDITOR_AGENT_VERBOSE, LLM_TIMEOUT

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _generate_storyboard(content: str, store=None, previous=None) -> str:
    """Generate a storyboard based on the input content.

    Args:
        content (str): The input content for the storyboard.
        store (BaseStore, optional): The store for chat state. Defaults to None.
        previous (ChatState, optional): Previous chat state. Defaults to None.

    Returns:
        str: The generated storyboard.
    """
    try:
        prompt_template = config.prompts.get('storyboard_generation_prompt', '')
        formatted_prompt = prompt_template.format(
            recent_chat_history=previous.get_recent_history_str(),
            memories=previous.get_memories_str(),
            tool_results=previous.get_tool_results_str()
        )
        storyboard_result = formatted_prompt
        await process_storyboard_images(storyboard_result, message_id=previous.thread_id)
        return storyboard_result
    except Exception as e:
        cl_logger.error(f"Storyboard generation failed: {e}")
        return "Error generating storyboard."

@task
async def generate_storyboard(content: str, **kwargs) -> str:
    return await _generate_storyboard(content, **kwargs)

storyboard_editor_agent = generate_storyboard  # Expose the function as storyboard_editor_agent
