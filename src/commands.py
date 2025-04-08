import os
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

# During pytest collection, Chainlit is not running, so @cl.command registration causes KeyError.
# This replaces cl.command with a dummy decorator during tests to avoid errors.
if "PYTEST_CURRENT_TEST" in os.environ or "PYTEST_RUNNING" in os.environ:
    cl_logger.info("Skipping Chainlit command registration during test run.")

    def _noop_decorator(*args, **kwargs):
        """Dummy decorator to replace @cl.command during tests."""
        def wrapper(func):
            return func
        return wrapper

    cl.command = _noop_decorator


async def _handle_agent_response(state: ChatState, vector_store: VectorStore, response_messages: list[AIMessage]):
    """Helper to handle adding agent response to state and vector store."""
    if response_messages:
        ai_msg = response_messages[0]
        # Ensure message_id is in metadata before adding to state/store
        if ai_msg and hasattr(ai_msg, 'metadata') and ai_msg.metadata and "message_id" in ai_msg.metadata:
            state.messages.append(ai_msg)
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"])
            cl_logger.debug(f"Added agent response to state and vector store: {ai_msg.name} (ID: {ai_msg.metadata['message_id']})")
        elif ai_msg:
            state.messages.append(ai_msg)
            await vector_store.put(content=ai_msg.content)
            cl_logger.warning(f"Agent response for {ai_msg.name} added to state, but missing message_id in metadata. Added to vector store without ID.")
        else:
            cl_logger.warning("Agent response was empty or None.")
    else:
        cl_logger.warning("Agent did not return any response messages.")


@cl.command(name="roll", description="Roll dice (e.g., /roll 2d6 or /roll check perception)")
async def command_roll(query: str):
    """Handles the /roll command."""
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    user_msg = HumanMessage(content=f"/roll {query}", name="Player")
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content)

    cl_logger.info(f"Executing /roll command with query: {query}")
    response_messages = await dice_agent(state)

    await _handle_agent_response(state, vector_store, response_messages)

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

    user_msg = HumanMessage(content=f"/search {query}", name="Player")
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content)

    cl_logger.info(f"Executing /search command with query: {query}")
    response_messages = await web_search_agent(state)

    await _handle_agent_response(state, vector_store, response_messages)

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

    user_msg = HumanMessage(content=f"/todo {query}", name="Player")
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content)

    cl_logger.info(f"Executing /todo command with query: {query}")
    response_messages = await todo_agent(state)

    await _handle_agent_response(state, vector_store, response_messages)

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

    user_msg = HumanMessage(content=f"/write {query}", name="Player")
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content)

    cl_logger.info(f"Executing /write command with query: {query}")
    response_messages = await writer_agent(state)

    await _handle_agent_response(state, vector_store, response_messages)

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
            if hasattr(msg, 'metadata') and msg.metadata and "message_id" in msg.metadata:
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
