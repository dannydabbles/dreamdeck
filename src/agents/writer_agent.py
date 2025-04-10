from src.config import config
import os
import logging
import asyncio
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
import sys

# Make 'cl' accessible as an attribute of this module for monkeypatching in tests
sys.modules[__name__].cl = cl

# Initialize logging
cl_logger = logging.getLogger("chainlit")

class _WriterAgentWrapper:
    async def __call__(self, state: ChatState, **kwargs) -> list[BaseMessage]:
        """Makes the wrapper instance callable by LangGraph, delegating to the task."""
        return await generate_story(state, **kwargs)

writer_agent = _WriterAgentWrapper()


@cl.step(name="Writer Agent: Generate Story", type="tool")
async def _generate_story(state: ChatState) -> list[BaseMessage]:
    """Generate the Game Master's narrative response based on recent chat, memories, and tool results."""
    try:
        persona = state.current_persona
        cl_logger.info(f"Writer agent using persona: {persona}")

        # Determine the prompt key based on persona
        prompt_key = "default_writer_prompt"
        try:
            # Access nested config structure safely
            persona_configs = getattr(config.agents.writer_agent, "personas", {})
            if isinstance(persona_configs, dict):
                persona_entry = persona_configs.get(persona)
                if persona_entry and isinstance(persona_entry, dict):
                    prompt_key = persona_entry.get("prompt_key", prompt_key)
        except Exception:
            pass  # Defensive: fallback to default_writer_prompt

        cl_logger.info(f"Writer agent resolved prompt key: {prompt_key}")

        # Get the actual prompt template string using the key
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

        persona_icon = {
            "Therapist": "🧠",
            "Secretary": "🗒️",
            "Coder": "💻",
            "Friend": "🤝",
            "Lorekeeper": "📚",
            "Dungeon Master": "🎲",
            "Storyteller GM": "🎭",
            "Default": "🤖",
        }.get(state.current_persona, "🤖")

        story_segment = AIMessage(
            content=gm_message.content.strip(),
            name=f"{persona_icon} {state.current_persona}",
            metadata={"message_id": gm_message.id},
        )

        return [story_segment]
    except Exception as e:
        cl_logger.error(f"Story generation failed: {e}")
        return [AIMessage(content="Story generation failed.", name="error", metadata={"message_id": None})]


@task
async def generate_story(state: ChatState, **kwargs) -> list[BaseMessage]:
    result = await _generate_story(state)
    return result


# Expose the @task-decorated function as a separate callable
generate_story_task = generate_story

# Expose internal function for monkeypatching in tests
writer_agent._generate_story = _generate_story
writer_agent.generate_story = generate_story  # <-- Add this attribute for patching


async def call_writer_agent(state: ChatState) -> list[BaseMessage]:
    """Call the writer agent outside of LangGraph workflows (e.g., slash commands)."""
    return await _generate_story(state)


# Expose call_writer_agent on the wrapper for easier patching in tests
writer_agent.call_writer_agent = call_writer_agent
