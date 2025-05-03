"""
Slash command handlers for Dreamdeck.

These commands bypass the decision agent and directly invoke the relevant tool or agent.
They are registered as Chainlit actions and are available in the UI and slash menu.

All commands are also exposed as action buttons and in the Chainlit chat settings.
"""

import logging
import os
import sys

import chainlit as cl

# NOTE: Chainlit v2+ does not use cl.command or cl.set_commands.
# Commands and actions are registered via event_handlers.py using action_callback decorators and ChatSettings.
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.dice_agent import dice_agent
from src.agents.report_agent import report_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.todo_agent import call_todo_agent, todo_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.writer_agent import writer_agent
from src.config import IMAGE_GENERATION_ENABLED, START_MESSAGE
from src.models import ChatState
from src.stores import VectorStore

cl_logger = logging.getLogger("chainlit")


async def command_empty():
    """Handles the case where the slash command is just `/` with no command name."""
    await cl.Message(content="Unknown command: /").send()


async def command_roll(query: str = ""):
    """Slash command: /roll - Roll dice via natural language or dice notation

    Creates a user message, appends it to state, calls the dice agent, and stores the AI response.
    """
    try:
        state: ChatState = cl.user_session.get("state")
        vector_store: VectorStore = cl.user_session.get("vector_memory")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not state:
            cl_logger.error(f"Command /roll: ChatState not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Session state not found. Cannot process command.").send()
            return
        if not vector_store:
            cl_logger.error(f"Command /roll: VectorStore not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Vector memory not found. Cannot process command.").send()
            return

        cl_logger.info(f"Executing /roll command for thread {state.thread_id} with query: {query}")

    except Exception as e:
        cl_logger.error(f"Command /roll: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(
        content=f"/roll {query}",
        author="Player",
        actions=[
            cl.Action(id="roll_again", name="Roll Again", payload={}, type="button"),
            cl.Action(
                id="continue_story", name="Continue Story", payload={}, type="button"
            ),
        ],
    )
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(
        content=f"/roll {query}", name="Player", metadata={"message_id": user_cl_msg_id}
    )
    state.messages.append(user_msg)
    await vector_store.put(
        content=user_msg.content,
        message_id=user_cl_msg_id,
        metadata={
            "type": "human",
            "author": "Player",
            "persona": state.current_persona,
        },
    )

    cl_logger.info(f"Executing /roll command with query: {query}")
    response_messages = await dice_agent(state)

    from src.storage import append_log

    append_log(state.current_persona, "Tool call: /roll")

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(
                content=ai_msg.content,
                message_id=ai_msg.metadata["message_id"],
                metadata={
                    "type": "ai",
                    "author": ai_msg.name,
                    "persona": state.current_persona,
                },
            )
        else:
            cl_logger.warning(
                f"AIMessage from dice_agent missing message_id: {ai_msg.content}"
            )

    cl.user_session.set("state", state)
    cl_logger.info(f"/roll command processed.")


async def command_search(query: str = ""):
    """Slash command: /search - Perform a web search

    Creates a user message, appends it to state, calls the web search agent, and stores the AI response.
    """
    try:
        state: ChatState = cl.user_session.get("state")
        vector_store: VectorStore = cl.user_session.get("vector_memory")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not state:
            cl_logger.error(f"Command /search: ChatState not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Session state not found. Cannot process command.").send()
            return
        if not vector_store:
            cl_logger.error(f"Command /search: VectorStore not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Vector memory not found. Cannot process command.").send()
            return

        cl_logger.info(f"Executing /search command for thread {state.thread_id} with query: {query}")

    except Exception as e:
        cl_logger.error(f"Command /search: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(content=f"/search {query}", author="Player")
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(
        content=f"/search {query}",
        name="Player",
        metadata={"message_id": user_cl_msg_id},
    )
    state.messages.append(user_msg)
    await vector_store.put(
        content=user_msg.content,
        message_id=user_cl_msg_id,
        metadata={
            "type": "human",
            "author": "Player",
            "persona": state.current_persona,
        },
    )

    cl_logger.info(f"Executing /search command with query: {query}")
    response_messages = await web_search_agent(state)

    from src.storage import append_log

    append_log(state.current_persona, "Tool call: /search")

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(
                content=ai_msg.content,
                message_id=ai_msg.metadata["message_id"],
                metadata={
                    "type": "ai",
                    "author": ai_msg.name,
                    "persona": state.current_persona,
                },
            )
        else:
            cl_logger.warning(
                f"AIMessage from web_search_agent missing message_id: {ai_msg.content}"
            )

    cl.user_session.set("state", state)
    cl_logger.info(f"/search command processed.")


async def command_todo(query: str = ""):
    """Slash command: /todo - Add a TODO item

    Creates a user message, appends it to state, calls the todo agent, and stores the AI response.
    """
    try:
        state: ChatState = cl.user_session.get("state")
        vector_store: VectorStore = cl.user_session.get("vector_memory")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not state:
            cl_logger.error(f"Command /todo: ChatState not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Session state not found. Cannot process command.").send()
            return
        if not vector_store:
            cl_logger.error(f"Command /todo: VectorStore not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Vector memory not found. Cannot process command.").send()
            return

        cl_logger.info(f"Executing /todo command for thread {state.thread_id} with query: {query}")

    except Exception as e:
        cl_logger.error(f"Command /todo: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(content=f"/todo {query}", author="Player")
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(
        content=f"/todo {query}", name="Player", metadata={"message_id": user_cl_msg_id}
    )
    state.messages.append(user_msg)
    await vector_store.put(
        content=user_msg.content,
        message_id=user_cl_msg_id,
        metadata={
            "type": "human",
            "author": "Player",
            "persona": state.current_persona,
        },
    )

    cl_logger.info(f"Executing /todo command with query: {query}")
    response_messages = await call_todo_agent(state)

    from src.storage import append_log

    append_log(state.current_persona, "Tool call: /todo")

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        # Defensive: skip metadata check if error message (which lacks metadata)
        if (
            hasattr(ai_msg, "metadata")
            and ai_msg.metadata
            and "message_id" in ai_msg.metadata
        ):
            await vector_store.put(
                content=ai_msg.content,
                message_id=ai_msg.metadata["message_id"],
                metadata={
                    "type": "ai",
                    "author": ai_msg.name,
                    "persona": state.current_persona,
                },
            )
        else:
            cl_logger.warning(
                f"AIMessage from todo_agent missing message_id: {ai_msg.content}"
            )

    cl.user_session.set("state", state)
    cl_logger.info(f"/todo command processed.")


async def command_write(query: str = ""):
    """Slash command: /write - Directly prompt the writer agent (current persona)

    Creates a user message, appends it to state, calls the writer agent, and stores the AI response.
    """
    try:
        state: ChatState = cl.user_session.get("state")
        vector_store: VectorStore = cl.user_session.get("vector_memory")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not state:
            cl_logger.error(f"Command /write: ChatState not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Session state not found. Cannot process command.").send()
            return
        if not vector_store:
            cl_logger.error(f"Command /write: VectorStore not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Vector memory not found. Cannot process command.").send()
            return

        cl_logger.info(f"Executing /write command for thread {state.thread_id} with query: {query}")

    except Exception as e:
        cl_logger.error(f"Command /write: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data.").send()
        return

    # Create and send user message
    user_cl_msg = cl.Message(content=f"/write {query}", author="Player")
    await user_cl_msg.send()
    user_cl_msg_id = user_cl_msg.id

    # Update state and vector store for user message
    user_msg = HumanMessage(
        content=f"/write {query}",
        name="Player",
        metadata={"message_id": user_cl_msg_id},
    )
    state.messages.append(user_msg)
    await vector_store.put(
        content=user_msg.content,
        message_id=user_cl_msg_id,
        metadata={
            "type": "human",
            "author": "Player",
            "persona": state.current_persona,
        },
    )

    cl_logger.info(f"Executing /write command with query: {query}")
    from src.agents.writer_agent import call_writer_agent

    response_messages = await call_writer_agent(state)

    from src.storage import append_log

    append_log(state.current_persona, "Tool call: /write")

    # Update state and vector store for AI response
    if response_messages:
        ai_msg = response_messages[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(
                content=ai_msg.content,
                message_id=ai_msg.metadata["message_id"],
                metadata={
                    "type": "ai",
                    "author": ai_msg.name,
                    "persona": state.current_persona,
                },
            )
        else:
            cl_logger.warning(
                f"AIMessage from writer_agent missing message_id: {ai_msg.content}"
            )

    cl.user_session.set("state", state)
    cl_logger.info(f"/write command processed.")


async def command_storyboard(query: str = ""):
    """Slash command: /storyboard - Generate storyboard images for the last Game Master message

    Finds the last Game Master message, then calls the storyboard editor agent to generate images.
    """
    try:
        state: ChatState = cl.user_session.get("state")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not state:
            cl_logger.error(f"Command /storyboard: ChatState not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Session state not found. Cannot process command.").send()
            return

        cl_logger.info(f"Executing /storyboard command for thread {state.thread_id}")

    except Exception as e:
        cl_logger.error(f"Command /storyboard: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data.").send()
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
                cl_logger.warning(
                    f"Found last GM message, but it's missing message_id in metadata: {msg.content[:50]}..."
                )

    if last_gm_message_id:
        cl_logger.info(
            f"Executing /storyboard command for message ID: {last_gm_message_id}"
        )
        await cl.Message(content="Generating storyboard for the last scene...").send()
        await storyboard_editor_agent(state=state, gm_message_id=last_gm_message_id)
        from src.storage import append_log

        append_log(state.current_persona, "Tool call: /storyboard")
        cl_logger.info(f"/storyboard command completed for thread {state.thread_id}.")
    else:
        await cl.Message(
            content="Could not find a previous Game Master message with a valid ID to generate a storyboard for."
        ).send()
        cl_logger.warning(
            "Could not execute /storyboard: No suitable GM message found in state."
        )


async def command_help():
    """Slash command: /help - Show help message"""
    help_text = """
**Available Commands:**

/roll [dice or description] — Roll dice (e.g., `/roll 2d6` or `/roll check perception`)
/search [query] — Perform a web search
/todo [note] — Add a TODO item
/write [prompt] — Directly prompt the writer agent
/storyboard — Generate storyboard images for the last Game Master message
/report — Generate a daily summary report
/persona [name] — Force switch to a specific persona
/help — Show this help message
/reset — Reset the current story and start fresh
/save — Export the current story as a markdown file
"""
    await cl.Message(content=help_text.strip()).send()


async def command_reset():
    """Slash command: /reset - Reset the current story"""
    try:
        vector_store: VectorStore = cl.user_session.get("vector_memory")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not vector_store:
            # Log error but allow reset to continue partially (clearing state)
            cl_logger.error(f"Command /reset: VectorStore not found in session for thread {thread_id}. Vector store cannot be cleared.")
            # await cl.Message(content="Error: Vector memory not found. Cannot clear vector store.").send() # Optional user message

        cl_logger.info(f"Executing /reset command for thread {thread_id}")

    except Exception as e:
        cl_logger.error(f"Command /reset: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data during reset.").send()
        return

    cl_logger.info(f"Resetting chat state and vector store for thread {thread_id}")
    # Clear state
    state = ChatState(messages=[], thread_id=cl.context.session.thread_id)
    cl.user_session.set("state", state)

    # Clear vector store collection (only if found)
    if vector_store:
        try:
            # Get the collection name from the vector store instance if possible, otherwise use thread_id
            collection_name = getattr(vector_store.collection, 'name', thread_id)
            cl_logger.info(f"Attempting to delete non-knowledge data from Chroma collection: {collection_name}")
            # Assuming delete works by filtering metadata, adjust if Chroma API differs
            await vector_store.collection.delete(where={"type": {"$ne": "knowledge"}})
            cl_logger.info(f"Cleared non-knowledge data from vector store collection: {collection_name}")
        except AttributeError:
             cl_logger.warning(f"Vector store object does not have a 'collection' attribute or 'delete' method. Skipping vector store clear.")
        except Exception as e:
            cl_logger.error(f"Failed to clear vector store for thread {thread_id}: {e}", exc_info=True)
            # Optionally inform the user
            # await cl.Message(content="Warning: Failed to clear vector memory.").send()

    # Send fresh start message
    start_msg = cl.Message(content=START_MESSAGE, author="Game Master")
    await start_msg.send()
    state.messages.append(
        AIMessage(
            content=START_MESSAGE,
            name="Game Master",
            metadata={"message_id": start_msg.id},
        )
    )
    cl.user_session.set("state", state)
    cl_logger.info(f"/reset command processed for thread {thread_id}.")


async def command_save():
    """Slash command: /save - Export the current story as markdown"""
    try:
        state: ChatState = cl.user_session.get("state")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not state:
            cl_logger.error(f"Command /save: ChatState not found in session for thread {thread_id}.")
            await cl.Message(content="No story to save.").send()
            return

        cl_logger.info(f"Executing /save command for thread {state.thread_id}")

    except Exception as e:
        cl_logger.error(f"Command /save: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data for saving.").send()
        return

    def escape_md(text):
        return text.replace("```", "\\`\\`\\`")

    md_lines = []
    for msg in state.messages:
        if isinstance(msg, HumanMessage):
            md_lines.append(f"**Player:** {escape_md(msg.content)}")
        elif isinstance(msg, AIMessage):
            md_lines.append(f"**{msg.name or 'AI'}:** {escape_md(msg.content)}")

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
    cl_logger.info(f"/save command processed for thread {state.thread_id}.")


async def command_report():
    """Slash command: /report - Generate a daily summary report"""
    try:
        state: ChatState = cl.user_session.get("state")
        vector_store: VectorStore = cl.user_session.get("vector_memory")
        thread_id = cl.context.session.thread_id if cl.context else "Unknown"

        if not state:
            cl_logger.error(f"Command /report: ChatState not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Session state not found. Cannot process command.").send()
            return
        if not vector_store:
            cl_logger.error(f"Command /report: VectorStore not found in session for thread {thread_id}.")
            await cl.Message(content="Error: Vector memory not found. Cannot process command.").send()
            return

        cl_logger.info(f"Executing /report command for thread {state.thread_id}")

    except Exception as e:
        cl_logger.error(f"Command /report: Error accessing session data: {e}", exc_info=True)
        await cl.Message(content="Error: Failed to access session data.").send()
        return

    responses = await report_agent(state)

    from src.storage import append_log

    append_log(state.current_persona, "Tool call: /report")

    if responses:
        ai_msg = responses[0]
        state.messages.append(ai_msg)
        if ai_msg.metadata and "message_id" in ai_msg.metadata:
            await vector_store.put(
                content=ai_msg.content,
                message_id=ai_msg.metadata["message_id"],
                metadata={"type": "ai", "author": ai_msg.name},
            )
    cl_user_session = cl.user_session
    cl_user_session.set("state", state)
    cl_logger.info(f"/report command processed for thread {state.thread_id}.")


async def command_persona(query: str = ""):
    """Slash command: /persona [name] - Force switch persona immediately"""
    try:
        state: ChatState = cl.user_session.get("state") # Still useful for logging thread_id
        thread_id = state.thread_id if state else (cl.context.session.thread_id if cl.context else "Unknown")
        cl_logger.info(f"Executing /persona command for thread {thread_id} with query: {query}")
    except Exception as e:
        cl_logger.error(f"Command /persona: Error accessing session data: {e}", exc_info=True)
        # Don't return here, as the command might still work partially

    from src.config import config  # Fix: import config here to avoid F821
    persona_name = query.strip()
    if not persona_name:
        await cl.Message(content="Usage: `/persona [persona_name]`").send()
        return

    valid_personas = list(config.agents.writer_agent.personas.keys()) + ["Default"]
    if persona_name.lower() not in [p.lower() for p in valid_personas]:
        await cl.Message(
            content=f"⚠️ Unknown persona '{persona_name}'. Valid options: {', '.join(valid_personas)}"
        ).send()
        return

    # Update session and state
    cl.user_session.set("current_persona", persona_name)
    state: ChatState = cl.user_session.get("state")
    if state:
        state = state.model_copy(update={"current_persona": persona_name})
        cl.user_session.set("state", state)

    from src.storage import append_log

    append_log(
        persona_name, f"Persona forcibly switched to {persona_name} via slash command."
    )

    cl_logger.info(f"Persona forcibly switched to: {persona_name} via slash command for thread {thread_id}")
    await cl.Message(
        content=f"✅ Persona forcibly switched to **{persona_name}**."
    ).send()
