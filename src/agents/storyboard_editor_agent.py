from src.config import config
import asyncio
import logging
from jinja2 import Template
import random  # Import random
from chainlit import Message as CLMessage  # Import CLMessage from Chainlit
from chainlit import Image as CLImage  # Import Image from Chainlit
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from src.image_generation import generate_image_async, generate_image_generation_prompts
from src.config import (
    STORYBOARD_EDITOR_AGENT_TEMPERATURE,
    STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
    STORYBOARD_EDITOR_AGENT_STREAMING,
    STORYBOARD_EDITOR_AGENT_VERBOSE,
    LLM_TIMEOUT,
    STORYBOARD_GENERATION_PROMPT,
)
from src.models import ChatState


# Initialize logging
cl_logger = logging.getLogger("chainlit")


async def _generate_storyboard(
    state: ChatState, gm_message_id: str
) -> list[BaseMessage]:
    """Generate a storyboard based on the input content.
    content = state.messages[-1].content if state.messages else ""

    Args:
        content (str): The input content for the storyboard.
        store (BaseStore, optional): The store for chat state. Defaults to None.
        previous (ChatState, optional): Previous chat state. Defaults to None.

    Returns:
        str: The generated storyboard.
    """
    messages = state.messages
    try:
        # Format STORYBOARD_GENERATION_PROMPT as jinja2
        template = Template(STORYBOARD_GENERATION_PROMPT)
        formatted_prompt = template.render(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
        )

        # Initialize the LLM
        llm = ChatOpenAI(
            base_url="http://192.168.1.111:5000/v1",
            temperature=STORYBOARD_EDITOR_AGENT_TEMPERATURE,
            max_tokens=STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
            streaming=STORYBOARD_EDITOR_AGENT_STREAMING,
            verbose=STORYBOARD_EDITOR_AGENT_VERBOSE,
            timeout=LLM_TIMEOUT,
        )

        # Generate the storyboard
        storyboard_response = await llm.ainvoke([("system", formatted_prompt)])
        storyboard = storyboard_response.content.strip()

        await process_storyboard_images(storyboard, message_id=gm_message_id)
        return []
    except Exception as e:
        cl_logger.error(f"Storyboard generation failed: {e}")
        return [AIMessage(content="Error generating storyboard.", name="error")]


@task
async def generate_storyboard(
    state: ChatState, gm_message_id: str
) -> list[BaseMessage]:
    asyncio.create_task(_generate_storyboard(state, gm_message_id))
    return []


async def process_storyboard_images(storyboard: str, message_id: str) -> None:
    """Process storyboard into images and send to chat.

    Args:
        storyboard (str): The storyboard content.
        message_id (str): The message ID for the chat.
    """
    if not storyboard or not config.features.image_generation:
        return  # Early exit if no content

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


storyboard_editor_agent = (
    generate_storyboard  # Expose the function as storyboard_editor_agent
)
