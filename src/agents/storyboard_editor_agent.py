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
    STORYBOARD_EDITOR_AGENT_BASE_URL,
    LLM_TIMEOUT,
    STORYBOARD_GENERATION_PROMPT,
)
from src.models import ChatState

import chainlit as cl

# Initialize logging
cl_logger = logging.getLogger("chainlit")


@cl.step(name="Storyboard Editor Agent", type="tool")
async def _generate_storyboard(
    state: ChatState, gm_message_id: str, **kwargs
) -> list[BaseMessage]:
    """Generate a storyboard prompt from recent chat, then asynchronously generate images."""
    messages = state.messages
    try:
        # Format STORYBOARD_GENERATION_PROMPT as jinja2
        template = Template(STORYBOARD_GENERATION_PROMPT)
        formatted_prompt = template.render(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
            user_preferences=state.user_preferences,
        )

        # Get user settings and defaults
        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get(
            "storyboard_temp", STORYBOARD_EDITOR_AGENT_TEMPERATURE
        )
        final_endpoint = (
            user_settings.get("storyboard_endpoint") or STORYBOARD_EDITOR_AGENT_BASE_URL
        )
        final_max_tokens = user_settings.get(
            "storyboard_max_tokens", STORYBOARD_EDITOR_AGENT_MAX_TOKENS
        )

        # Initialize the LLM with potentially overridden settings
        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
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
        return [
            AIMessage(
                content="Error generating storyboard.",
                name="error",
                metadata={"message_id": None},
            )
        ]


@task
async def generate_storyboard(
    state: ChatState, gm_message_id: str, **kwargs
) -> list[BaseMessage]:
    return await _generate_storyboard(state, gm_message_id, **kwargs)


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


# Refactored: storyboard_editor_agent is now a stateless, LLM-backed function (task)
@task
async def storyboard_editor_agent(state: ChatState, gm_message_id: str, **kwargs) -> list[BaseMessage]:
    return await _generate_storyboard(state, gm_message_id, **kwargs)

# Helper for non-langgraph context (slash commands, CLI, etc)
async def storyboard_editor_agent_helper(state: ChatState, gm_message_id: str, **kwargs) -> list[BaseMessage]:
    return await _generate_storyboard(state, gm_message_id, **kwargs)
