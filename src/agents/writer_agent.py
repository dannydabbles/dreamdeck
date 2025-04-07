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


async def _generate_story(state: ChatState) -> list[BaseMessage]:
    """Generate a story segment based on the input content.
    content = state.messages[-1].content if state.messages else ""
    store = state.vector_memory
    previous = state

    Args:
        content (str): The input content for the story.
        store (BaseStore, optional): The store for chat state. Defaults to None.
        previous (ChatState, optional): Previous chat state. Defaults to None.

    Returns:
        str: The generated story segment.
    """
    try:
        # Format AI_WRITER_PROMPT as jinja2
        template = Template(AI_WRITER_PROMPT)
        formatted_prompt = template.render(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
        )

        # Initialize the LLM
        llm = ChatOpenAI(
            base_url=WRITER_AGENT_BASE_URL,
            temperature=WRITER_AGENT_TEMPERATURE,
            max_tokens=WRITER_AGENT_MAX_TOKENS,
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
            content=gm_message.content.strip(), name="Game Master"
        )

        return [story_segment]
    except Exception as e:
        cl_logger.error(f"Story generation failed: {e}")
        return []


@task
async def generate_story(state: ChatState, **kwargs) -> list[BaseMessage]:
    return await _generate_story(state)


writer_agent = generate_story  # Expose the function as writer_agent
