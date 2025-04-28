import asyncio
import logging
import random  # Import random

import chainlit as cl
from chainlit import Image as CLImage  # Import Image from Chainlit
from chainlit import Message as CLMessage  # Import CLMessage from Chainlit
from jinja2 import Template
from langchain_core.messages import AIMessage, BaseMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from langgraph.func import task
from langgraph.prebuilt import create_react_agent

from src.config import (
    LLM_TIMEOUT,
    STORYBOARD_EDITOR_AGENT_BASE_URL,
    STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
    STORYBOARD_EDITOR_AGENT_STREAMING,
    STORYBOARD_EDITOR_AGENT_TEMPERATURE,
    STORYBOARD_EDITOR_AGENT_VERBOSE,
    STORYBOARD_GENERATION_PROMPT,
    config,
)
from src.image_generation import generate_image_async, generate_image_generation_prompts
from src.models import ChatState

# Initialize logging
cl_logger = logging.getLogger("chainlit")


@cl.step(name="Storyboard Editor Agent", type="tool")
async def _generate_storyboard(
    state: ChatState, gm_message_id: str, sd_api_url: str = None, **kwargs
) -> list[BaseMessage]:
    """Generate a storyboard prompt from recent chat, then generate images."""
    try:
        cl_logger.info(f"Starting storyboard for message ID: {gm_message_id}")
        if sd_api_url:
            cl_logger.info(f"Using Stable Diffusion URL: {sd_api_url}")
        else:
            cl_logger.warning("No Stable Diffusion API URL provided to storyboard agent.")
        gm_message = next(
            msg for msg in state.messages 
            if isinstance(msg, AIMessage) 
            and msg.metadata.get("message_id") == gm_message_id
        )
        cl_logger.info(f"Found GM message: {gm_message.content[:50]}...")

        # Format prompt using the GM's message content
        template = Template(config.loaded_prompts["storyboard_generation_prompt"])
        formatted_prompt = template.render(
            recent_chat_history=gm_message.content,  # Use GM's message directly
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
            model=config.llm.model,
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
        cl_logger.info(f"Generated storyboard: {storyboard[:200]}...")

        # Process images after generating storyboard
        await process_storyboard_images(storyboard, gm_message_id, sd_api_url)
        
        return [
            AIMessage(
                content=f"🎨 Generated storyboard for: {gm_message.content[:50]}...",
                name="storyboard",
                metadata={"message_id": gm_message_id},
            )
        ]
    except StopIteration:
        cl_logger.error(f"No GM message found with ID: {gm_message_id}")
        return []
    except Exception as e:
        cl_logger.error("Storyboard failed", exc_info=True)
        return []


@task
async def generate_storyboard(
    state: ChatState, gm_message_id: str, **kwargs
) -> list[BaseMessage]:
    return await _generate_storyboard(state, gm_message_id, **kwargs)


async def process_storyboard_images(storyboard: str, message_id: str, sd_api_url: str = None) -> None:
    if not storyboard or not config.features.image_generation:
        return

    try:
        image_prompts = await generate_image_generation_prompts(storyboard)
        for prompt in image_prompts:
            image_bytes = await generate_image_async(prompt, random.randint(0, 2**32))
            if image_bytes:
                image_element = CLImage(
                    content=image_bytes,
                    display="inline",
                    size="large",
                )
                await CLMessage(
                    content=f"🎨 {prompt}",
                    elements=[image_element],
                    parent_id=message_id
                ).send()
    except Exception as e:
        cl_logger.error(f"Storyboard error: {str(e)}")


# Refactored: storyboard_editor_agent is now a stateless, LLM-backed function (task)
@task
async def storyboard_editor_agent(
    state: ChatState, gm_message_id: str, **kwargs
) -> list[BaseMessage]:
    return await _generate_storyboard(state, gm_message_id, **kwargs)


# Helper for non-langgraph context (slash commands, CLI, etc)
async def storyboard_editor_agent_helper(
    state: ChatState, gm_message_id: str, **kwargs
) -> list[BaseMessage]:
    return await _generate_storyboard(state, gm_message_id, **kwargs)
