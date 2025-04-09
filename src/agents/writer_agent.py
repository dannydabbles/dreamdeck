from src.config import config
import os
import logging
from jinja2 import Template
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from src.config import (
    WRITER_AGENT_TEMPERATURE,
    WRITER_AGENT_MAX_TOKENS,
    WRITER_AGENT_STREAMING,
    WRITER_AGENT_VERBOSE,
    LLM_TIMEOUT,
    WRITER_AGENT_BASE_URL,
    AI_WRITER_PROMPT,
)
from src.models import ChatState

import chainlit as cl

# Initialize logging
cl_logger = logging.getLogger("chainlit")


@cl.step(name="Writer Agent: Generate Story", type="tool")
async def _generate_story(state: ChatState) -> list[BaseMessage]:
    """Generate the Game Master's narrative response based on recent chat, memories, and tool results."""
    try:
        persona = state.current_persona
        cl_logger.info(f"Writer agent using persona: {persona}")

        # Determine the prompt key based on persona
        # Access nested config structure carefully
        persona_config = config.agents.writer_agent.dict().get("personas", {}).get(persona, {})
        prompt_key = persona_config.get("prompt_key", "default_writer_prompt") # Fallback key

        # Get the actual prompt template string using the key
        # Fallback to the AI_WRITER_PROMPT constant if key not found in loaded_prompts
        prompt_template_str = config.loaded_prompts.get(prompt_key, AI_WRITER_PROMPT)

        # Format prompt as jinja2
        template = Template(prompt_template_str)
        formatted_prompt = template.render(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
            user_preferences=state.user_preferences,
        )

        # Get user settings and defaults
        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("writer_temp", WRITER_AGENT_TEMPERATURE)
        final_endpoint = user_settings.get("writer_endpoint") or WRITER_AGENT_BASE_URL
        final_max_tokens = user_settings.get("writer_max_tokens", WRITER_AGENT_MAX_TOKENS)

        # Initialize the LLM with potentially overridden settings
        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=WRITER_AGENT_STREAMING,
            verbose=WRITER_AGENT_VERBOSE,
            timeout=LLM_TIMEOUT,
        )

        # Generate the story
        gm_message: cl.Message = cl.Message(content="")
        cl.user_session.set("gm_message", gm_message)

        cb = cl.AsyncLangchainCallbackHandler(
            to_ignore=[
                "ChannelRead",
                "RunnableLambda",
                "ChannelWrite",
                "__start__",
                "_execute",
            ],
        )

        async for chunk in llm.astream([("system", formatted_prompt)]):
            await gm_message.stream_token(chunk.content)
        await gm_message.send()

        story_segment = AIMessage(
            content=gm_message.content.strip(),
            name="Game Master",
            metadata={"message_id": gm_message.id},
        )

        return [story_segment]
    except Exception as e:
        cl_logger.error(f"Story generation failed: {e}")
        return [AIMessage(content="Story generation failed.", name="error", metadata={"message_id": None})]


@task
async def generate_story(state: ChatState, **kwargs) -> list[BaseMessage]:
    return await _generate_story(state)


writer_agent = generate_story  # Expose the function as writer_agent


async def call_writer_agent(state: ChatState) -> list[BaseMessage]:
    """Call the writer agent outside of LangGraph workflows (e.g., slash commands)."""
    return await _generate_story(state)
