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
from .agents.orchestrator_agent import orchestrator_agent
from .agents.writer_agent import writer_agent
from .agents.storyboard_editor_agent import storyboard_editor_agent
from src.config import IMAGE_GENERATION_ENABLED
from .models import ChatState
from langchain_core.stores import BaseStore

from src.agents import agents_map

import chainlit as cl
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
)

import logging

from src.config import MAX_CHAIN_LENGTH

cl_logger = logging.getLogger("chainlit")


@entrypoint(checkpointer=MemorySaver())
async def chat_workflow(
    inputs: dict,
) -> ChatState:
    messages = inputs.get("messages", [])
    previous = inputs.get("previous", None)

    cl_logger.info(f"Received {len(messages)} messages.")

    return await _chat_workflow(messages=messages, previous=previous)


async def _chat_workflow(
    messages: list[BaseMessage],
    previous: ChatState,
) -> ChatState:
    cl_logger.info(f"Messages: {messages}")
    cl_logger.info(f"Previous state: {previous}")

    new_messages = previous.messages.copy()
    new_messages.extend(messages)

    state = ChatState(
        messages=new_messages,
        thread_id=previous.thread_id,
        error_count=previous.error_count,
    )

    try:
        vector_memory = cl.user_session.get("vector_memory")

        chain_count = 0
        actions = await orchestrator_agent(state)
        cl_logger.info(f"Initial orchestrator actions: {actions}")

        while actions and chain_count < MAX_CHAIN_LENGTH:
            # Remove trailing GM call if present
            if actions[-1] in ("write", "continue_story"):
                gm_action = actions.pop()
                gm_needed = True
            else:
                gm_needed = False

            # Process tool actions
            for action in actions:
                agent_func = agents_map.get(action)
                if not agent_func:
                    cl_logger.warning(f"Unknown action from orchestrator: {action}")
                    continue

                agent_response = await agent_func(state)

                for msg in agent_response:
                    state.messages.append(msg)
                    if vector_memory and msg.metadata and "message_id" in msg.metadata:
                        await vector_memory.put(
                            content=msg.content,
                            message_id=msg.metadata["message_id"],
                            metadata={"type": "ai", "author": msg.name},
                        )

            # After tools, decide next actions
            chain_count += 1
            if gm_needed or chain_count >= MAX_CHAIN_LENGTH:
                # Call GM to continue story
                writer_response = await writer_agent(state)
                if writer_response:
                    for msg in writer_response:
                        state.messages.append(msg)
                        if vector_memory and msg.metadata and "message_id" in msg.metadata:
                            await vector_memory.put(
                                content=msg.content,
                                message_id=msg.metadata["message_id"],
                                metadata={"type": "ai", "author": msg.name},
                            )
                break  # Always stop after GM
            else:
                # Re-orchestrate based on updated state
                actions = await orchestrator_agent(state)
                cl_logger.info(f"Next orchestrator actions: {actions}")

        # Trigger storyboard generation after GM response
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
                content="The adventure continues...",
                name="system",
                metadata={}
            )
        )

    return state
