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
from .agents.director_agent import director_agent
from .agents.writer_agent import writer_agent
from .agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.knowledge_agent import knowledge_agent
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

from src.config import MAX_CHAIN_LENGTH, config

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
        actions = await director_agent(state)
        cl_logger.info(f"Initial director actions: {actions}")

        # Persona-aware filtering and reordering of actions (Phase 5)
        persona = getattr(state, "current_persona", "default").lower()
        prefs = config.PERSONA_TOOL_PREFERENCES.get(persona, {})

        avoid = set(prefs.get("avoid", []))
        prefer = prefs.get("prefer", [])

        # Filter out avoided actions
        filtered_actions = []
        for act in actions:
            # act can be string or dict
            if isinstance(act, str):
                if act in avoid:
                    cl_logger.info(f"Skipping action '{act}' due to persona '{persona}' avoid list")
                    continue
            elif isinstance(act, dict):
                # For dict actions, check 'action' key
                act_name = act.get("action")
                if act_name in avoid:
                    cl_logger.info(f"Skipping action '{act_name}' due to persona '{persona}' avoid list")
                    continue
            filtered_actions.append(act)

        # Reorder preferred actions to the front, preserving relative order
        preferred_actions = []
        other_actions = []
        for act in filtered_actions:
            act_name = act
            if isinstance(act, dict):
                act_name = act.get("action")
            if act_name in prefer:
                preferred_actions.append(act)
            else:
                other_actions.append(act)

        actions = preferred_actions + other_actions

        cl_logger.info(f"Persona-aware filtered and reordered actions: {actions}")

        while actions and chain_count < MAX_CHAIN_LENGTH:
            # Remove trailing GM call if present
            if actions[-1] in ("write", "continue_story"):
                gm_action = actions.pop()
                gm_needed = True
            else:
                gm_needed = False

            # Process tool actions
            for action in actions:
                agent_response = []
                if isinstance(action, str):
                    agent_func = agents_map.get(action)
                    if not agent_func:
                        cl_logger.warning(f"Unknown action string from director: {action}")
                        continue
                    agent_response = await agent_func(state)
                elif isinstance(action, dict):
                    action_name = action.get("action")
                    if action_name == "knowledge":
                        knowledge_type = action.get("type")
                        if knowledge_type:
                            agent_response = await knowledge_agent(state, knowledge_type=knowledge_type)
                        else:
                            cl_logger.warning(f"Knowledge action missing 'type': {action}")
                            continue

                for msg in agent_response:
                    state.messages.append(msg)
                    if vector_memory and msg.metadata and "message_id" in msg.metadata:
                        await vector_memory.put(
                            content=msg.content,
                            message_id=msg.metadata["message_id"],
                            metadata={"type": "ai", "author": msg.name, "persona": state.current_persona},
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
                                metadata={"type": "ai", "author": msg.name, "persona": state.current_persona},
                            )
                break  # Always stop after GM
            else:
                # Re-orchestrate based on updated state
                actions = await director_agent(state)
                cl_logger.info(f"Next director actions: {actions}")

                # Persona-aware filtering and reordering of actions (Phase 5)
                persona = getattr(state, "current_persona", "default").lower()
                prefs = config.PERSONA_TOOL_PREFERENCES.get(persona, {})

                avoid = set(prefs.get("avoid", []))
                prefer = prefs.get("prefer", [])

                # Filter out avoided actions
                filtered_actions = []
                for act in actions:
                    # act can be string or dict
                    if isinstance(act, str):
                        if act in avoid:
                            cl_logger.info(f"Skipping action '{act}' due to persona '{persona}' avoid list")
                            continue
                    elif isinstance(act, dict):
                        # For dict actions, check 'action' key
                        act_name = act.get("action")
                        if act_name in avoid:
                            cl_logger.info(f"Skipping action '{act_name}' due to persona '{persona}' avoid list")
                            continue
                    filtered_actions.append(act)

                # Reorder preferred actions to the front, preserving relative order
                preferred_actions = []
                other_actions = []
                for act in filtered_actions:
                    act_name = act
                    if isinstance(act, dict):
                        act_name = act.get("action")
                    if act_name in prefer:
                        preferred_actions.append(act)
                    else:
                        other_actions.append(act)

                actions = preferred_actions + other_actions

                cl_logger.info(f"Persona-aware filtered and reordered actions: {actions}")

        # Trigger storyboard generation after GM response
        if IMAGE_GENERATION_ENABLED:
            last_ai_message = None
            if state.messages and isinstance(state.messages[-1], AIMessage):
                last_ai_message = state.messages[-1]

            if last_ai_message and last_ai_message.metadata and "message_id" in last_ai_message.metadata:
                gm_message_id = last_ai_message.metadata["message_id"]
                cl_logger.info(f"Triggering storyboard generation for message ID: {gm_message_id}")
                asyncio.create_task(storyboard_editor_agent(state=state, gm_message_id=gm_message_id))
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
