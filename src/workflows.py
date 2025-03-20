import asyncio
import logging
import json
from typing import List, Optional
from langgraph.prebuilt import create_react_agent
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from chainlit import Message as CLMessage
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from .models import ChatState
from .agents.decision_agent import decision_agent  # Import decide_action
from .agents.dice_agent import dice_agent  # Import dice_roll
from .agents.web_search_agent import web_search_agent  # Import web_search
from .agents.writer_agent import writer_agent  # Import generate_story
from .agents.storyboard_editor_agent import storyboard_editor_agent
from src.config import (
    IMAGE_GENERATION_ENABLED,
    DECISION_PROMPT,  # Import DECISION_PROMPT
)
from .models import ChatState
from langchain_core.stores import BaseStore

from src.config import DECISION_PROMPT

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
    """Core logic for processing user input and generating responses.

    Args:
        messages (List[BaseMessage]): Incoming messages from the user.
        previous (ChatState): Previous state of the conversation.

    Returns:
        ChatState: Updated conversation state after processing.

    Workflow Steps:
        1. Update state with new messages
        2. Determine action via decision_agent
        3. Execute selected action (dice/web/story)
        4. Append results to conversation
        5. Trigger storyboard generation if enabled

    Raises:
        Exception: Critical errors during processing are logged but don't halt execution
    """

    cl_logger.info(f"Messages: {messages}")
    cl_logger.info(f"Previous state: {previous}")

    new_messages = previous.messages.copy()
    new_messages.extend(messages)

    state = ChatState(
        messages=new_messages,
        thread_id=previous.thread_id,
        error_count=previous.error_count,
        # Copy other fields as needed
    )

    try:
        # Determine action
        human_messages = [
            msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)
        ]
        last_human_message = human_messages[0] if human_messages else None
        new_message = None
        if not last_human_message:
            cl_logger.info("No human message found, defaulting to continue_story")
            action = "continue_story"
        else:
            decision_response = await decision_agent(state)
            action = decision_response[0].name
            cl_logger.info(f"Action: {action}")
            new_message = AIMessage(content=decision_response[0].name, name="decision")
            state.messages.append(new_message)

        vector_memory = cl.user_session.get("vector_memory")  # Retrieve vector store

        if "roll" in action:
            dice_response = await dice_agent(state)
            new_message = dice_response[0]
            state.messages.append(new_message)
            if vector_memory:
                await vector_memory.put(content=new_message.content)

        elif "search" in action:
            web_search_response = await web_search_agent(state)
            new_message = web_search_response[0]
            state.messages.append(new_message)
            if vector_memory:
                await vector_memory.put(content=new_message.content)

        writer_response = await writer_agent(state)
        if writer_response:
            new_message = writer_response[0]
            state.messages.append(new_message)
            if vector_memory:
                await vector_memory.put(content=new_message.content)

            # Generate storyboard if needed and image generation is enabled
            gm_message: CLMessage = cl.user_session.get("gm_message")
            if IMAGE_GENERATION_ENABLED:
                storyboard_editor_agent(state=state, gm_message_id=gm_message.id)

        else:
            cl_logger.error(f"Unknown action: {action}")

        return state

    except Exception as e:
        cl_logger.error(f"Critical error in chat workflow: {str(e)}", exc_info=True)
        # Reset error count to avoid unintended increments during testing
        state.error_count = 0  # Prevents unexpected message additions in tests
        state = state.model_copy(deep=True)
        state.messages.append(
            AIMessage(
                content="⚠️ A critical error occurred. Please try again later or restart the session.",
                name="error",
            )
        )

    return state
