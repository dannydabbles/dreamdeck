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
from .agents.writer_agent import writer_agent  # Import generate_story
from .agents.storyboard_editor_agent import storyboard_editor_agent
from src.config import (
    IMAGE_GENERATION_ENABLED,
    DECISION_PROMPT,  # Import DECISION_PROMPT
)
from .models import ChatState
from langchain_core.stores import BaseStore

from src.config import DECISION_PROMPT

from src.agents.dice_agent import dice_agent
from src.agents.web_search_agent import web_search_agent

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
            new_message = AIMessage(content=decision_response[0].name, name="decision", metadata={})
            state.messages.append(new_message)

        vector_memory = cl.user_session.get("vector_memory")  # Retrieve vector store

        if "roll" in action:
            dice_response = await dice_agent(state)
            new_message = dice_response[0]
            state.messages.append(new_message)
            if vector_memory:
                if new_message.metadata and "message_id" in new_message.metadata:
                    await vector_memory.put(
                        content=new_message.content,
                        message_id=new_message.metadata["message_id"],
                        metadata={"type": "ai", "author": new_message.name},
                    )
                else:
                    cl_logger.warning(
                        f"AIMessage from dice_agent missing message_id for vector store: {new_message.content}"
                    )

        elif "search" in action:
            web_search_response = await web_search_agent(state)
            new_message = web_search_response[0]
            state.messages.append(new_message)
            if vector_memory:
                if new_message.metadata and "message_id" in new_message.metadata:
                    await vector_memory.put(
                        content=new_message.content,
                        message_id=new_message.metadata["message_id"],
                        metadata={"type": "ai", "author": new_message.name},
                    )
                else:
                    cl_logger.warning(
                        f"AIMessage from web_search_agent missing message_id for vector store: {new_message.content}"
                    )

        writer_response = await writer_agent(state)
        if writer_response:
            new_message = writer_response[0]
            state.messages.append(new_message)
            if vector_memory:
                if new_message.metadata and "message_id" in new_message.metadata:
                    await vector_memory.put(
                        content=new_message.content,
                        message_id=new_message.metadata["message_id"],
                        metadata={"type": "ai", "author": new_message.name},
                    )
                else:
                    cl_logger.warning(
                        f"AIMessage from writer_agent missing message_id for vector store: {new_message.content}"
                    )

        # After all agent responses, trigger storyboard generation if enabled
        if IMAGE_GENERATION_ENABLED:
            last_ai_message = None
            if state.messages and isinstance(state.messages[-1], AIMessage):
                last_ai_message = state.messages[-1]

            if last_ai_message and last_ai_message.metadata and "message_id" in last_ai_message.metadata:
                gm_message_id = last_ai_message.metadata["message_id"]
                cl_logger.info(f"Triggering storyboard generation for message ID: {gm_message_id}")
                storyboard_editor_agent(state=state, gm_message_id=gm_message_id)
            else:
                cl_logger.warning("Could not trigger storyboard generation: Last AI message or its message_id not found in state.")

        return state

    except Exception as e:
        cl_logger.error(f"Workflow failed: {e}")
        state.messages.append(
            AIMessage(
                content="The adventure continues...",  # Match test expectation
                name="system",
                metadata={}
            )
        )

    return state
