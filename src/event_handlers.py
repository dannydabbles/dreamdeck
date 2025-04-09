import os
import logging
import asyncio
import random
import base64
import httpx
from typing import List, Optional

# Define Chainlit commands for UI buttons
commands = [
    {"id": "roll", "icon": "dice-5", "description": "Roll dice"},
    {"id": "search", "icon": "globe", "description": "Web search"},
    {"id": "todo", "icon": "list", "description": "Add a TODO"},
    {"id": "write", "icon": "pen-line", "description": "Direct prompt to writer"},
    {"id": "storyboard", "icon": "image", "description": "Generate storyboard"},
    {"id": "help", "icon": "help-circle", "description": "Show help"},
    {"id": "reset", "icon": "refresh-ccw", "description": "Reset story"},
    {"id": "save", "icon": "save", "description": "Export story"},
]
from chainlit.types import ThreadDict
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    FunctionMessage,
    ToolMessage,
)
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableConfig  # Import RunnableConfig
from src.config import (
    NEGATIVE_PROMPT,
    STEPS,
    SAMPLER_NAME,
    SCHEDULER,
    CFG_SCALE,
    WIDTH,
    HEIGHT,
    HR_UPSCALER,
    DENOISING_STRENGTH,
    HR_SECOND_PASS_STEPS,
    IMAGE_GENERATION_TIMEOUT,
    REFUSAL_LIST,
    KNOWLEDGE_DIRECTORY,
    STORYBOARD_GENERATION_PROMPT_PREFIX,
    STORYBOARD_GENERATION_PROMPT_POSTFIX,
    AI_WRITER_PROMPT,
    IMAGE_GENERATION_ENABLED,
    WEB_SEARCH_ENABLED,
    DICE_ROLLING_ENABLED,
    START_MESSAGE,
)
from src.initialization import init_db, DatabasePool
from src.models import ChatState
from src.workflows import chat_workflow
from src.initialization import DatabasePool  # Import DatabasePool

from src.stores import VectorStore  # Import VectorStore
from src.agents.decision_agent import decision_agent
from src.agents.writer_agent import writer_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.dice_agent import dice_roll_agent, dice_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.todo_agent import todo_agent
from chainlit import user_session as cl_user_session  # Import cl_user_session
from langchain_core.callbacks.manager import CallbackManagerForChainRun  # Import CallbackManagerForChainRun

from langchain_core.stores import BaseStore

import chainlit as cl
from chainlit.input_widget import Slider, TextInput  # Import widgets
from src import config  # Import your config
from chainlit import Action  # Import Action for buttons

# Centralized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chainlit.log")],
)

cl_logger = logging.getLogger("chainlit")


# Define an asynchronous range generator
async def async_range(end):
    """Asynchronous range generator.

    Args:
        end (int): The end value for the range.
    """
    for i in range(0, end):
        # Sleep for a short duration to simulate asynchronous operation
        await asyncio.sleep(0.1)
        yield i


def _load_document(file_path):
    if file_path.endswith(".pdf"):
        return PyMuPDFLoader(file_path).load()
    elif file_path.endswith(".txt"):
        return TextLoader(file_path).load()
    elif file_path.endswith(".md"):
        return UnstructuredMarkdownLoader(file_path).load()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    elif (username, password) == ("test", "test"):
        return cl.User(
            identifier="test", metadata={"role": "test", "provider": "credentials"}
        )
    elif (username, password) == ("guest", "guest"):
        return cl.User(
            identifier="guest", metadata={"role": "guest", "provider": "credentials"}
        )
    else:
        return None


@cl.on_chat_start
async def on_chat_start():
    """Initialize new chat session with Chainlit integration.

    Sets up the user session, initializes the chat state, initializes agents and vector store,
    and sends initial messages. Agents and vector store are stored in the Chainlit user session.
    """

    # Determine active persona from profile selection, default if none
    persona = cl.user_session.get("current_persona", "Default")

    try:
        # Fetch current user identifier
        user_info = cl.user_session.get("user")
        if isinstance(user_info, dict):
            current_user_identifier = user_info.get("identifier", "Player")
        elif hasattr(user_info, 'identifier'):
            current_user_identifier = user_info.identifier
        else:
            current_user_identifier = "Player"

        # Initialize vector store
        vector_memory = VectorStore()
        cl.user_session.set("vector_memory", vector_memory)

        # Initialize agents
        cl_user_session.set("decision_agent", decision_agent)
        cl_user_session.set("writer_agent", writer_agent)
        cl_user_session.set("storyboard_editor_agent", storyboard_editor_agent)
        cl_user_session.set("dice_roll_agent", dice_roll_agent)
        cl_user_session.set("web_search_agent", web_search_agent)

        # Register Chainlit commands for UI buttons
        await cl.context.emitter.set_commands(commands)

        # Define Chat Settings
        settings = await cl.ChatSettings(
            [
                Slider(
                    id="writer_temp",
                    label="Writer Agent - Temperature",
                    min=0.0, max=2.0, step=0.1, initial=config.WRITER_AGENT_TEMPERATURE
                ),
                Slider(
                    id="writer_max_tokens",
                    label="Writer Agent - Max Tokens",
                    min=100, max=16000, step=100, initial=config.WRITER_AGENT_MAX_TOKENS
                ),
                TextInput(
                    id="writer_endpoint",
                    label="Writer Agent - OpenAI Endpoint URL",
                    initial=config.WRITER_AGENT_BASE_URL or "",
                    placeholder="e.g., http://localhost:5000/v1"
                ),
            ]
        ).send()

        # Persist initial chat settings in user session
        cl.user_session.set("chat_settings", settings)

        # Launch knowledge loading in the background
        asyncio.create_task(load_knowledge_documents())

        # Initialize thread in Chainlit with a start message
        start_cl_msg = cl.Message(
            content=START_MESSAGE,
            author=current_user_identifier,
        )
        await start_cl_msg.send()

        # Create initial state
        state = ChatState(
            messages=[AIMessage(content=START_MESSAGE, name="Game Master", metadata={"message_id": start_cl_msg.id})],
            thread_id=cl.context.session.thread_id,
            user_preferences=cl.user_session.get("user_session", {}).get(
                "preferences", {}
            ),
            current_persona=persona, # Set persona in initial state
        )

        # Store state
        cl.user_session.set("state", state)
        cl.user_session.set("image_generation_memory", [])
        cl_user_session.set("ai_message_id", None)

    except Exception as e:
        cl_logger.error(f"Application failed to start: {e}", exc_info=True)
        raise
    finally:
        # Close database pool
        await DatabasePool.close()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread.

    Initializes agents and vector store, reconstructs chat state from thread history,
    and stores them in the Chainlit user session.
    """
    # Determine persona from thread tags, default if none found
    resumed_persona = "Default"
    tags = thread.get("tags", [])
    for tag in tags:
        if tag.startswith("chainlit-profile:"):
            resumed_persona = tag.split(":", 1)[1]
            cl_logger.info(f"Resumed persona '{resumed_persona}' from thread tags.")
            break
    cl.user_session.set("current_persona", resumed_persona) # Also set in session for consistency
    # Set the user in the session
    user_dict = thread.get("user")
    if user_dict:
        cl.user_session.set("user", user_dict)

    # Fetch current user identifier
    user_info = cl.user_session.get("user")
    if isinstance(user_info, dict):
        current_user_identifier = user_info.get("identifier", "Player")
    elif hasattr(user_info, 'identifier'):
        current_user_identifier = user_info.identifier
    else:
        current_user_identifier = "Player"

    # Initialize vector store
    vector_memory = VectorStore()
    cl.user_session.set("vector_memory", vector_memory)

    # Initialize thread in Chainlit with a start message
    messages = []
    image_generation_memory = []
    cl.user_session.set("gm_message", cl.Message(content="", author=current_user_identifier))

    # Reconstruct messages from thread history
    for step in sorted(thread.get("steps", []), key=lambda m: m.get("createdAt", "")):
        step_id = step.get("id")
        parent_id = step.get("parentId")
        meta = {"message_id": step_id}
        if parent_id:
            meta["parent_id"] = parent_id

        # Check if message already exists in vector store to avoid duplicates
        existing = False
        try:
            res = vector_memory.collection.get(ids=[step_id])
            if res and res.get("ids") and step_id in res["ids"]:
                existing = True
        except Exception:
            pass

        if step["type"] == "user_message":
            cl_msg = cl.Message(
                content=step["output"],
                author=current_user_identifier,
            )
            # await cl_msg.send()  # Disabled to avoid duplicate UI messages
            messages.append(HumanMessage(content=step["output"], name="Player", metadata=meta))
            if step_id and not existing:
                meta = {"type": "human", "author": "Player"}
                if parent_id is not None:
                    meta["parent_id"] = parent_id
                await vector_memory.put(content=step["output"], message_id=step_id, metadata=meta)
            elif not step_id:
                cl_logger.warning(f"Missing ID for user step in on_chat_resume: {step.get('output', '')[:50]}...")
        elif step["type"] == "assistant_message":
            cl_msg = cl.Message(
                content=step["output"],
                author=current_user_identifier,
            )
            # await cl_msg.send()  # Disabled to avoid duplicate UI messages
            messages.append(AIMessage(content=step["output"], name=step["name"], metadata=meta))
            if step_id and not existing:
                meta = {"type": "ai", "author": step.get("name", "Unknown")}
                if parent_id is not None:
                    meta["parent_id"] = parent_id
                await vector_memory.put(content=step["output"], message_id=step_id, metadata=meta)
            elif not step_id:
                cl_logger.warning(f"Missing ID for assistant step in on_chat_resume: {step.get('output', '')[:50]}...")

    # Create state
    state = ChatState(
        messages=messages,
        thread_id=thread["id"],
        user_preferences=cl.user_session.get("user_session", {}).get("preferences", {}),
        thread_data=cl.user_session.get("thread_data", {}),
        current_persona=resumed_persona, # Set persona in resumed state
    )

    # Store state and memories
    cl.user_session.set("state", state)
    cl_user_session.set("image_generation_memory", image_generation_memory)
    cl_user_session.set("ai_message_id", None)

    # Load knowledge documents
    await load_knowledge_documents()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user chat messages.

    Ignores slash commands (handled separately).
    Adds user message to state and vector store.
    Calls the chat workflow, which invokes the decision agent and relevant tools.
    """
    state: ChatState = cl.user_session.get("state")
    vector_memory: VectorStore = cl.user_session.get("vector_memory")

    # Retrieve current user identifier from session
    current_user_identifier = None
    user_info = cl.user_session.get("user")
    if user_info:
        if isinstance(user_info, dict):
            current_user_identifier = user_info.get("identifier")
        elif hasattr(user_info, 'identifier'):
            current_user_identifier = user_info.identifier

    is_current_user = current_user_identifier and message.author == current_user_identifier
    is_generic_player = message.author == "Player"

    if is_current_user or is_generic_player:
        if not is_current_user and is_generic_player:
            cl_logger.warning(f"Processing message from 'Player' author, but couldn't verify against session identifier '{current_user_identifier}'.")

        # If the message is a command button click, handle it
        if message.command is not None and message.command != "":
            cl_logger.info(f"Command button selected: {message.command}")
            try:
                from src import commands as cmd_mod
                cmd_name = message.command.lower()
                arg = ""  # No argument from button click
                if cmd_name == "roll":
                    await cmd_mod.command_roll(arg)
                elif cmd_name == "search":
                    await cmd_mod.command_search(arg)
                elif cmd_name == "todo":
                    await cmd_mod.command_todo(arg)
                elif cmd_name == "write":
                    await cmd_mod.command_write(arg)
                elif cmd_name == "storyboard":
                    await cmd_mod.command_storyboard(arg)
                elif cmd_name == "help":
                    await cmd_mod.command_help()
                elif cmd_name == "reset":
                    await cmd_mod.command_reset()
                elif cmd_name == "save":
                    await cmd_mod.command_save()
                else:
                    await cl.Message(content=f"Unknown command: {cmd_name}").send()
            except Exception as e:
                cl_logger.error(f"Error handling command button '{message.command}': {e}", exc_info=True)
                await cl.Message(content=f"Error processing command '{message.command}': {e}").send()
            return  # Skip normal message processing

        # If the message is a slash command, parse and dispatch
        if message.content.strip().startswith("/"):
            command_line = message.content.strip()
            parts = command_line.split(maxsplit=1)
            command = parts[0][1:].lower()
            arg = parts[1] if len(parts) > 1 else ""

            cl_logger.info(f"Slash command detected: /{command} with arg: '{arg}'")

            try:
                from src import commands  # Import here to avoid circular imports
                if command == "roll":
                    await commands.command_roll(arg)
                elif command == "search":
                    await commands.command_search(arg)
                elif command == "todo":
                    await commands.command_todo(arg)
                elif command == "write":
                    await commands.command_write(arg)
                elif command == "storyboard":
                    await commands.command_storyboard(arg)
                elif command == "help":
                    await commands.command_help()
                elif command == "reset":
                    await commands.command_reset()
                elif command == "save":
                    await commands.command_save()
                else:
                    await cl.Message(content=f"Unknown command: /{command}").send()
            except Exception as e:
                cl_logger.error(f"Error handling slash command /{command}: {e}", exc_info=True)
                await cl.Message(content=f"Error processing command /{command}: {e}").send()

            return  # Skip normal message processing

        # Add user message to state immediately
        user_msg = HumanMessage(content=message.content, name="Player", metadata={"message_id": message.id})
        state.messages.append(user_msg)
        # Add user message to vector memory
        await vector_memory.put(content=message.content, message_id=message.id, metadata={"type": "human", "author": "Player"})

        try:
            state.memories = [
                str(m.page_content) for m in vector_memory.get(message.content)
            ]
            cl_logger.info(f"Memories: {state.memories}")

            thread_config = {
                "configurable": {
                    "thread_id": state.thread_id,
                }
            }

            cb = cl.AsyncLangchainCallbackHandler(
                to_ignore=[
                    "ChannelRead",
                    "RunnableLambda",
                    "ChannelWrite",
                    "__start__",
                    "_execute",
                ],
            )

            inputs = {"messages": state.messages, "previous": state}
            state = await chat_workflow.ainvoke(
                inputs, config=RunnableConfig(callbacks=[cb], **thread_config)
            )

            if state.messages and isinstance(state.messages[-1], AIMessage):
                ai_msg = state.messages[-1]
                if ai_msg.metadata and "message_id" in ai_msg.metadata:
                    await vector_memory.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name})
                else:
                    cl_logger.warning(f"Final AIMessage from workflow missing message_id: {ai_msg.content}")

            cl.user_session.set("state", state)

        except Exception as e:
            cl_logger.error(f"Runnable stream failed: {e}", exc_info=True)
            cl_logger.error(f"State: {state}")
            await cl.Message(
                content="⚠️ An error occurred while generating the response. Please try again later.",
            ).send()
            return
    else:
        cl_logger.debug(f"Ignoring message from author '{message.author}'. Expected identifier: '{current_user_identifier}'.")


async def load_knowledge_documents():
    """Load documents from the knowledge directory into the vector store."""
    if not os.path.exists(KNOWLEDGE_DIRECTORY):
        cl.element.logger.warning(
            f"Knowledge directory '{KNOWLEDGE_DIRECTORY}' does not exist. Skipping document loading."
        )
        return

    vector_memory = cl.user_session.get("vector_memory", None)

    if not vector_memory:
        cl.element.logger.error("Vector memory not initialized.")
        return

    documents = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    loop = asyncio.get_event_loop()

    for root, dirs, files in os.walk(KNOWLEDGE_DIRECTORY):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Offload CPU-heavy work to a thread
                loaded_docs = await loop.run_in_executor(
                    None, _load_document, file_path
                )
                # Tag knowledge docs with metadata
                for doc in loaded_docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["type"] = "knowledge"
                    doc.metadata["source"] = file_path

                split_docs = await loop.run_in_executor(
                    None, text_splitter.split_documents, loaded_docs
                )
                # Also tag split docs
                for doc in split_docs:
                    doc.metadata = doc.metadata or {}
                    doc.metadata["type"] = "knowledge"
                    doc.metadata["source"] = file_path

                documents.extend(split_docs)

                # Periodically flush to vector store to prevent memory bloat
                if len(documents) >= 500:
                    await vector_memory.add_documents(documents)
                    documents = []
            except Exception as e:
                cl.element.logger.error(f"Error processing {file_path}: {e}")

    # Final flush of remaining documents
    if documents:
        await vector_memory.add_documents(documents)



@cl.on_settings_update
async def on_settings_update(settings):
    cl.user_session.set("chat_settings", settings)
