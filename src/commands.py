import os
import sys
import chainlit as cl
import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState
from src.stores import VectorStore
from src.agents import (
    dice_agent,
    web_search_agent,
    todo_agent,
    writer_agent,
    storyboard_editor_agent,
)
from src.config import IMAGE_GENERATION_ENABLED

cl_logger = logging.getLogger("chainlit")

# Patch cl.command with a dummy decorator during *any* pytest phase (collection or run)
if (
    "PYTEST_CURRENT_TEST" in os.environ
    or "PYTEST_RUNNING" in os.environ
    or "pytest" in sys.modules
    or any("pytest" in arg for arg in sys.argv)
):
    cl_logger.info("Patching Chainlit command decorator during pytest collection/run.")

    def _noop_decorator(*args, **kwargs):
        """Dummy decorator to replace @cl.command during tests."""
        def wrapper(func):
            return func
        return wrapper

    cl.command = _noop_decorator


@cl.command(name="roll", description="Roll dice (e.g., /roll 2d6 or /roll check perception)")
async def command_roll(query: str):
    """Handles the /roll command."""
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(content=f"/roll {query}", author="Player")
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/roll {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player"})

    cl_logger.info(f"Executing /roll command with query: {query}")
    response_messages = await dice_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name})
        else:
            cl_logger.warning(f"AIMessage from dice_agent missing message_id: {ai_msg.content}")

    cl.user_session.set("state", state)
    cl_logger.info(f"/roll command processed.")


@cl.command(name="search", description="Perform a web search (e.g., /search history of dragons)")
async def command_search(query: str):
    """Handles the /search command."""
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(content=f"/search {query}", author="Player")
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/search {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player"})

    cl_logger.info(f"Executing /search command with query: {query}")
    response_messages = await web_search_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name})
        else:
            cl_logger.warning(f"AIMessage from web_search_agent missing message_id: {ai_msg.content}")

    cl.user_session.set("state", state)
    cl_logger.info(f"/search command processed.")


@cl.command(name="todo", description="Add a TODO item (e.g., /todo Remember to buy milk)")
async def command_todo(query: str):
    """Handles the /todo command."""
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(content=f"/todo {query}", author="Player")
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/todo {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player"})

    cl_logger.info(f"Executing /todo command with query: {query}")
    response_messages = await todo_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name})
        else:
            cl_logger.warning(f"AIMessage from todo_agent missing message_id: {ai_msg.content}")

    cl.user_session.set("state", state)
    cl_logger.info(f"/todo command processed.")


@cl.command(name="write", description="Directly prompt the writer agent (e.g., /write The wizard casts a spell)")
async def command_write(query: str):
    """Handles the /write command."""
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(content=f"/write {query}", author="Player")
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/write {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player"})

    cl_logger.info(f"Executing /write command with query: {query}")
    response_messages = await writer_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name})
        else:
            cl_logger.warning(f"AIMessage from writer_agent missing message_id: {ai_msg.content}")

    cl.user_session.set("state", state)
    cl_logger.info(f"/write command processed.")


@cl.command(name="storyboard", description="Generate storyboard for the last scene")
async def command_storyboard(query: str = ""):
    """Handles the /storyboard command."""
    state: ChatState = cl.user_session.get("state")
    if not state:
        await cl.Message(content="Error: Session state not found.").send()
        return

    if not IMAGE_GENERATION_ENABLED:
        await cl.Message(content="Image generation is disabled.").send()
        return

    last_gm_message_id = None
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and msg.name == "Game Master":
            if msg.metadata and "message_id" in msg.metadata:
                last_gm_message_id = msg.metadata["message_id"]
                break
            else:
                cl_logger.warning(f"Found last GM message, but it's missing message_id in metadata: {msg.content[:50]}...")

    if last_gm_message_id:
        cl_logger.info(f"Executing /storyboard command for message ID: {last_gm_message_id}")
        await cl.Message(content="Generating storyboard for the last scene...").send()
        await storyboard_editor_agent(state=state, gm_message_id=last_gm_message_id)
        cl_logger.info(f"/storyboard command processed.")
    else:
        await cl.Message(content="Could not find a previous Game Master message with a valid ID to generate a storyboard for.").send()
        cl_logger.warning("Could not execute /storyboard: No suitable GM message found in state.")
