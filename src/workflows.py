import asyncio
import logging
import json
from typing import List, Optional
from langgraph.prebuilt import create_react_agent
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from .models import ChatState
from .agents.decision_agent import decision_agent  # Import decide_action
from .agents.dice_agent import dice_agent  # Import dice_roll
from .agents.web_search_agent import web_search_agent  # Import web_search
from .agents.writer_agent import writer_agent  # Import generate_story
from .agents.storyboard_editor_agent import storyboard_editor_agent
from .config import (
    IMAGE_GENERATION_ENABLED,
    DECISION_PROMPT,  # Import DECISION_PROMPT
)
from .models import ChatState
from langchain_core.stores import BaseStore

from .config import DECISION_PROMPT

import chainlit as cl


# Initialize logging
cl_logger = logging.getLogger("chainlit")

@entrypoint(checkpointer=MemorySaver())
async def chat_workflow(
    inputs: dict,
) -> ChatState:
    """Main chat workflow handling messages and state.

    Args:
        messages (List[BaseMessage]): List of incoming messages.
        previous (Optional[ChatState], optional): Previous chat state. Defaults to None.

    Returns:
        ChatState: The updated chat state.
    """
    messages = inputs.get("messages", [])
    previous = inputs.get("previous", None)

    cl_logger.info(f"Received {len(messages)} messages.")

    # Call _chat_workflow with the correct arguments
    return await _chat_workflow(messages=messages, previous=previous)

async def _chat_workflow(
    messages: List[BaseMessage],
    previous: ChatState,
) -> ChatState:
    
    cl_logger.info(f"Messages: {messages}")
    cl_logger.info(f"Previous state: {previous}")

    state = previous
    state.messages.extend(messages)

    try:
        # Determine action
        human_messages = [msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)]
        last_human_message = human_messages[0] if human_messages else None
        if not last_human_message:
            cl_logger.info("No human message found, defaulting to continue_story")
            action = "continue_story"
        else:
            decision_response = await decision_agent(state)
            action = decision_response[0].name
            cl_logger.info(f"Action: {action}")

        if "roll" in action:
            state.messages += dice_agent(state).result()

        elif "search" in action:
            state.messages += web_search_agent(state).result()

        elif action in ["continue_story", "writer"]:
            state.messages += writer_agent(state).result()
            last_human_message = [msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)][0]

            # Generate storyboard if needed and image generation is enabled
            if IMAGE_GENERATION_ENABLED:
                storyboard_result = storyboard_editor_agent(state=state).result()
                if storyboard_result:
                    state.metadata["storyboard"] = storyboard_result

        else:
            cl_logger.error(f"Unknown action: {action}")

        return state

    except Exception as e:
        cl_logger.error(f"Critical error in chat workflow: {str(e)}", exc_info=True)
        state.increment_error_count()
        state.messages.append(AIMessage(content="⚠️ A critical error occurred. Please try again later or restart the session."))

    return state


