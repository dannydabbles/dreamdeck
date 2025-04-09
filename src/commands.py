"""
Slash command handlers for Dreamdeck.

These commands bypass the decision agent and directly invoke the relevant tool or agent.
"""

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
from src.agents.todo_agent import call_todo_agent
from src.config import IMAGE_GENERATION_ENABLED, START_MESSAGE

cl_logger = logging.getLogger("chainlit")


async def command_roll(query: str):
    """Handles the /roll command.

    Creates a user message, appends it to state, calls the dice agent, and stores the AI response.
    """
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(
        content=f"/roll {query}",
        author="Player"
    )
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/roll {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player", "persona": state.current_persona})

    cl_logger.info(f"Executing /roll command with query: {query}")
    response_messages = await dice_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name, "persona": state.current_persona})
        else:
            cl_logger.warning(f"AIMessage from dice_agent missing message_id: {ai_msg.content}")

    # Immediately call writer agent to continue story
    from src.agents.writer_agent import call_writer_agent
    gm_responses = await call_writer_agent(state)
    if gm_responses:
        gm_msg = gm_responses[0]
        state.messages.append(gm_msg)
        if gm_msg.metadata and "message_id" in gm_msg.metadata:
            await vector_store.put(content=gm_msg.content, message_id=gm_msg.metadata["message_id"], metadata={"type": "ai", "author": gm_msg.name})

    cl.user_session.set("state", state)
    cl_logger.info(f"/roll command processed.")


async def command_search(query: str):
    """Handles the /search command.

    Creates a user message, appends it to state, calls the web search agent, and stores the AI response.
    """
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(
        content=f"/search {query}",
        author="Player"
    )
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/search {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player", "persona": state.current_persona})

    cl_logger.info(f"Executing /search command with query: {query}")
    response_messages = await web_search_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name, "persona": state.current_persona})
        else:
            cl_logger.warning(f"AIMessage from web_search_agent missing message_id: {ai_msg.content}")

    # Immediately call writer agent to continue story
    from src.agents.writer_agent import call_writer_agent
    gm_responses = await call_writer_agent(state)
    if gm_responses:
        gm_msg = gm_responses[0]
        state.messages.append(gm_msg)
        if gm_msg.metadata and "message_id" in gm_msg.metadata:
            await vector_store.put(content=gm_msg.content, message_id=gm_msg.metadata["message_id"], metadata={"type": "ai", "author": gm_msg.name})

    cl.user_session.set("state", state)
    cl_logger.info(f"/search command processed.")


async def command_todo(query: str):
    """Handles the /todo command.

    Creates a user message, appends it to state, calls the todo agent, and stores the AI response.
    """
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(
        content=f"/todo {query}",
        author="Player"
    )
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/todo {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player", "persona": state.current_persona})

    cl_logger.info(f"Executing /todo command with query: {query}")
    response_messages = await call_todo_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        # Defensive: skip metadata check if error message (which lacks metadata)
        if hasattr(ai_msg, "metadata") and ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name, "persona": state.current_persona})
        else:
            cl_logger.warning(f"AIMessage from todo_agent missing message_id: {ai_msg.content}")

        # Send the AI response as a Chainlit message
        await cl.Message(content=str(ai_msg.content or "")).send()

    # Immediately call writer agent to continue story
    from src.agents.writer_agent import call_writer_agent
    gm_responses = await call_writer_agent(state)
    if gm_responses:
        gm_msg = gm_responses[0]
        state.messages.append(gm_msg)
        if gm_msg.metadata and "message_id" in gm_msg.metadata:
            await vector_store.put(content=gm_msg.content, message_id=gm_msg.metadata["message_id"], metadata={"type": "ai", "author": gm_msg.name})

    cl.user_session.set("state", state)
    cl_logger.info(f"/todo command processed.")


async def command_write(query: str):
    """Handles the /write command.

    Creates a user message, appends it to state, calls the writer agent, and stores the AI response.
    """
    state: ChatState = cl.user_session.get("state")
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if not state or not vector_store:
        await cl.Message(content="Error: Session state not found.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(
        content=f"/write {query}",
        author="Player"
    )
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(content=f"/write {query}", name="Player", metadata={"message_id": user_cl_msg_id})
    state.messages.append(user_msg)
    await vector_store.put(content=user_msg.content, message_id=user_cl_msg_id, metadata={"type": "human", "author": "Player", "persona": state.current_persona})

    cl_logger.info(f"Executing /write command with query: {query}")
    from src.agents.writer_agent import call_writer_agent
    response_messages = await call_writer_agent(state)

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name, "persona": state.current_persona})
        else:
            cl_logger.warning(f"AIMessage from writer_agent missing message_id: {ai_msg.content}")

    cl.user_session.set("state", state)
    cl_logger.info(f"/write command processed.")


async def command_storyboard(query: str = ""):
    """Handles the /storyboard command.

    Finds the last Game Master message, then calls the storyboard editor agent to generate images.
    """
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
async def command_help():
    help_text = """
**Available Commands:**

/roll [dice or description] — Roll dice (e.g., `/roll 2d6` or `/roll check perception`)
/search [query] — Perform a web search
/todo [note] — Add a TODO item
/write [prompt] — Directly prompt the writer agent
/storyboard — Generate storyboard images for the last Game Master message
/help — Show this help message
/reset — Reset the current story and start fresh
/save — Export the current story as a markdown file
"""
    await cl.Message(content=help_text.strip()).send()


async def command_reset():
    cl_logger.info("Resetting chat state and vector store")
    # Clear state
    state = ChatState(messages=[], thread_id=cl.context.session.thread_id)
    cl.user_session.set("state", state)

    # Clear vector store collection
    vector_store: VectorStore = cl.user_session.get("vector_memory")
    if vector_store:
        try:
            await vector_store.collection.delete(where={})
        except Exception:
            pass

    # Send fresh start message
    start_msg = cl.Message(content=START_MESSAGE, author="Game Master")
    await start_msg.send()
    state.messages.append(
        AIMessage(content=START_MESSAGE, name="Game Master", metadata={"message_id": start_msg.id})
    )
    cl.user_session.set("state", state)


async def command_save():
    state: ChatState = cl.user_session.get("state")
    if not state:
        await cl.Message(content="No story to save.").send()
        return

    md_lines = []
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            md_lines.append(f"**Player:** {msg.content}")
        elif isinstance(msg, AIMessage):
            md_lines.append(f"**{msg.name or 'AI'}:** {msg.content}")

    md_content = "\n\n".join(md_lines)
    # Send as downloadable file element
    await cl.Message(
        content="Here is your story so far:",
        elements=[
            cl.File(
                name="story.md",
                content=md_content.encode("utf-8"),
                mime="text/markdown",
            )
        ],
    ).send()
