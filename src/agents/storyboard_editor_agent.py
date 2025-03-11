from src.config import config
import os
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from src.image_generation import generate_image_async, generate_image_generation_prompts
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

async def process_storyboard_images(storyboard: str, message_id: str) -> None:
    """Process storyboard into images and send to chat.

    Args:
        storyboard (str): The storyboard content.
        message_id (str): The message ID for the chat.
    """
    if not storyboard or not config.features.image_generation:
        return

    try:
        # Generate image prompts
        image_prompts = await generate_image_generation_prompts(storyboard)

        # Process each prompt in order
        for prompt in image_prompts:
            try:
                # Generate image
                seed = random.randint(0, 2**32)
                image_bytes = await generate_image_async(prompt, seed)

                if image_bytes:
                    # Create and send image message
                    image_element = CLImage(
                        content=image_bytes,
                        display="inline",
                        size="large",
                        alt="Generated Image",
                        name="generated_image",
                    )

                    await CLMessage(
                        content=f"**Image Generation Prompt:**\n{prompt}\n\n**Seed:**\n{seed}",
                        elements=[image_element],
                        parent_id=message_id,
                    ).send()

            except Exception as e:
                cl_logger.error(
                    f"Failed to generate image for prompt: {prompt}. Error: {str(e)}"
                )

    except Exception as e:
        cl_logger.error(f"Failed to process storyboard images: {str(e)}")

storyboard_editor_agent = generate_storyboard  # Expose the function as storyboard_editor_agent

storyboard_editor_agent = generate_storyboard  # Expose the function as storyboard_editor_agent
