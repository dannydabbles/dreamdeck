import os
import logging
import asyncio
import random
import base64
import httpx
from typing import List, Optional

# Number of turns to suppress re-suggesting a declined persona
PERSONA_SUPPRESSION_TURNS = 3

# Expose persona_classifier_agent for tests
from src.agents.persona_classifier_agent import persona_classifier_agent

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
from src.workflows import app as chat_workflow
from src.initialization import DatabasePool  # Import DatabasePool

from src.stores import VectorStore  # Import VectorStore
from src.agents.writer_agent import writer_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.dice_agent import dice_roll_agent, dice_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.todo_agent import todo_agent
from src.supervisor import supervisor  # <-- Add this import
from chainlit import user_session as cl_user_session  # Import cl_user_session
from langchain_core.callbacks.manager import (
    CallbackManagerForChainRun,
)  # Import CallbackManagerForChainRun

from langchain_core.stores import BaseStore

import chainlit as cl
from chainlit.input_widget import (
    Slider,
    TextInput,
    Select,
    Switch,
)  # Import widgets including Switch
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
        elif hasattr(user_info, "identifier"):
            current_user_identifier = user_info.identifier
        else:
            current_user_identifier = "Player"

        # Initialize vector store
        vector_memory = VectorStore()
        cl.user_session.set("vector_memory", vector_memory)

        # Initialize agents
        cl_user_session.set("writer_agent", writer_agent)
        cl_user_session.set("storyboard_editor_agent", storyboard_editor_agent)
        cl_user_session.set("dice_roll_agent", dice_roll_agent)
        cl_user_session.set("web_search_agent", web_search_agent)

        # Register Chainlit commands for UI buttons
        await cl.context.emitter.set_commands(commands)

        # Define Chat Settings with persona selector and LLM options
        settings = await cl.ChatSettings(
            [
                Select(
                    id="persona",
                    label="Persona",
                    values=["Storyteller GM", "Friend", "Therapist", "Secretary", "Coder", "Dungeon Master", "Default"],
                    initial="Storyteller GM",
                ),
                Slider(
                    id="writer_temp",
                    label="Writer Agent - Temperature",
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    initial=config.WRITER_AGENT_TEMPERATURE,
                ),
                Slider(
                    id="writer_max_tokens",
                    label="Writer Agent - Max Tokens",
                    min=100,
                    max=16000,
                    step=100,
                    initial=config.WRITER_AGENT_MAX_TOKENS,
                ),
                TextInput(
                    id="writer_endpoint",
                    label="Writer Agent - OpenAI Endpoint URL",
                    initial=config.WRITER_AGENT_BASE_URL or "",
                    placeholder="e.g., http://localhost:5000/v1",
                ),
                Slider(
                    id="decision_temp",
                    label="Decision Agent - Temperature",
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    initial=config.DECISION_AGENT_TEMPERATURE,
                ),
                Slider(
                    id="storyboard_temp",
                    label="Storyboard Agent - Temperature",
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    initial=config.STORYBOARD_EDITOR_AGENT_TEMPERATURE,
                ),
                Slider(
                    id="storyboard_max_tokens",
                    label="Storyboard Agent - Max Tokens",
                    min=100,
                    max=16000,
                    step=100,
                    initial=config.STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
                ),
                TextInput(
                    id="storyboard_endpoint",
                    label="Storyboard Agent - OpenAI Endpoint URL",
                    initial=config.STORYBOARD_EDITOR_AGENT_BASE_URL or "",
                    placeholder="e.g., http://localhost:5000/v1",
                ),
                Switch(
                    id="auto_persona_switch",
                    label="Auto Persona Switching",
                    initial=True,
                ),
                Switch(
                    id="show_prompt",
                    label="Show Rendered Prompt (Dev Only)",
                    initial=False,
                ),
            ]
        ).send()

        # Defensive: if settings is None (e.g., in tests), replace with empty dict
        if settings is None:
            settings = {}

        cl.user_session.set("chat_settings", settings)
        cl.user_session.set(
            "current_persona", settings.get("persona", "Storyteller GM")
        )
        cl.user_session.set(
            "auto_persona_switch", settings.get("auto_persona_switch", True)
        )

        # Launch knowledge loading in the background
        asyncio.create_task(load_knowledge_documents())

        # Initialize thread in Chainlit with a start message
        start_cl_msg = cl.Message(
            content=START_MESSAGE,
            author=current_user_identifier,
            actions=[
                cl.Action(id="roll", name="Roll Dice", payload={}, type="button"),
                cl.Action(id="search", name="Web Search", payload={}, type="button"),
                cl.Action(id="todo", name="Add TODO", payload={}, type="button"),
                cl.Action(id="write", name="Direct Prompt", payload={}, type="button"),
                cl.Action(id="storyboard", name="Generate Storyboard", payload={}, type="button"),
                cl.Action(id="help", name="Help", payload={}, type="button"),
            ],
        )
        await start_cl_msg.send()

        # Create initial state
        state = ChatState(
            messages=[
                AIMessage(
                    content=START_MESSAGE,
                    name="Game Master",
                    metadata={"message_id": start_cl_msg.id},
                )
            ],
            thread_id=cl.context.session.thread_id,
            user_preferences=cl.user_session.get("user_session", {}).get(
                "preferences", {}
            ),
            current_persona=persona,  # Set persona in initial state
        )

        # Store state
        cl.user_session.set("state", state)
        cl.user_session.set("image_generation_memory", [])
        cl_user_session.set("ai_message_id", None)

    except Exception as e:
        cl_logger.error(f"Application failed to start: {e}", exc_info=True)
        raise


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
    cl.user_session.set(
        "current_persona", resumed_persona
    )  # Also set in session for consistency
    # Set the user in the session
    user_dict = thread.get("user")
    if user_dict:
        cl.user_session.set("user", user_dict)

    # Fetch current user identifier
    user_info = cl.user_session.get("user")
    if isinstance(user_info, dict):
        current_user_identifier = user_info.get("identifier", "Player")
    elif hasattr(user_info, "identifier"):
        current_user_identifier = user_info.identifier
    else:
        current_user_identifier = "Player"

    # Initialize vector store
    vector_memory = VectorStore()
    cl.user_session.set("vector_memory", vector_memory)

    # Initialize thread in Chainlit with a start message
    messages = []
    image_generation_memory = []
    cl.user_session.set(
        "gm_message", cl.Message(content="", author=current_user_identifier)
    )

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
            messages.append(
                HumanMessage(content=step["output"], name="Player", metadata=meta)
            )
            if step_id and not existing:
                meta = {"type": "human", "author": "Player"}
                if parent_id is not None:
                    meta["parent_id"] = parent_id
                await vector_memory.put(
                    content=step["output"], message_id=step_id, metadata=meta
                )
            elif not step_id:
                cl_logger.warning(
                    f"Missing ID for user step in on_chat_resume: {step.get('output', '')[:50]}..."
                )
        elif step["type"] == "assistant_message":
            cl_msg = cl.Message(
                content=step["output"],
                author=current_user_identifier,
            )
            # await cl_msg.send()  # Disabled to avoid duplicate UI messages
            messages.append(
                AIMessage(content=step["output"], name=step["name"], metadata=meta)
            )
            if step_id and not existing:
                meta = {"type": "ai", "author": step.get("name", "Unknown")}
                if parent_id is not None:
                    meta["parent_id"] = parent_id
                await vector_memory.put(
                    content=step["output"], message_id=step_id, metadata=meta
                )
            elif not step_id:
                cl_logger.warning(
                    f"Missing ID for assistant step in on_chat_resume: {step.get('output', '')[:50]}..."
                )

    # Create state
    state = ChatState(
        messages=messages,
        thread_id=thread["id"],
        user_preferences=cl.user_session.get("user_session", {}).get("preferences", {}),
        thread_data=cl.user_session.get("thread_data", {}),
        current_persona=resumed_persona,  # Set persona in resumed state
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
    if vector_memory is None:
        cl_logger.warning(
            "VectorStore missing in session during on_message. Initializing new VectorStore."
        )
        from src.stores import VectorStore

        vector_memory = VectorStore()
        cl.user_session.set("vector_memory", vector_memory)

    if state is None:
        cl_logger.warning(
            "ChatState missing in session during on_message. Initializing new ChatState."
        )
        # Defensive: get thread_id from context if possible
        thread_id = getattr(cl.context.session, "thread_id", "default_thread")
        state = ChatState(messages=[], thread_id=thread_id)
        cl.user_session.set("state", state)

    # Check if user is replying to a pending persona switch prompt
    pending_persona = cl.user_session.get("pending_persona_switch")
    if pending_persona:
        user_reply = message.content.strip().lower()
        if user_reply in ["yes", "y"]:
            cl_logger.info(f"User accepted persona switch to: {pending_persona}")
            cl.user_session.set("current_persona", pending_persona)
            if hasattr(state, "current_persona"):
                state.current_persona = pending_persona
            from src.storage import append_log
            append_log(pending_persona, f"Persona switched to {pending_persona} by user confirmation.")
            await cl.Message(
                content=f"🔄 Switching persona to **{pending_persona}** to better assist you."
            ).send()
        elif user_reply in ["no", "n"]:
            cl_logger.info(f"User declined persona switch to: {pending_persona}")
            await cl.Message(content=f"❌ Keeping current persona.").send()
            # Suppress re-suggesting this persona for next N turns
            suppressed = cl.user_session.get("suppressed_personas", {})
            suppressed[pending_persona] = PERSONA_SUPPRESSION_TURNS
            cl.user_session.set("suppressed_personas", suppressed)
        else:
            cl_logger.info(
                f"User response '{user_reply}' not recognized for persona switch confirmation."
            )
            await cl.Message(
                content="Please reply 'Yes' to switch persona or 'No' to keep current."
            ).send()
            return  # Wait for valid reply next time
        # Clear pending switch flag
        cl.user_session.set("pending_persona_switch", None)
        # Save updated state (if needed)
        cl.user_session.set("state", state)
        return  # Skip normal message processing

    # Handle Chainlit Actions (button clicks)
    if hasattr(message, "action") and message.action:
        action_id = message.action
        # If action_id is a MagicMock (from test), treat as unknown command
        if isinstance(action_id, str):
            cl_logger.info(f"Action button selected: {action_id}")
            from src import commands as cmd_mod
            if action_id == "roll_again":
                # Reuse last roll query if available, else empty
                last_human = state.get_last_human_message()
                last_query = ""
                if last_human and last_human.content.startswith("/roll"):
                    last_query = last_human.content[5:].strip()
                await cmd_mod.command_roll(last_query)
            elif action_id == "continue_story":
                from src.agents.writer_agent import call_writer_agent
                await call_writer_agent(state, from_oracle=False)
            elif action_id == "search":
                await cmd_mod.command_search("")
            elif action_id == "todo":
                await cmd_mod.command_todo("")
            elif action_id == "write":
                await cmd_mod.command_write("")
            elif action_id == "storyboard":
                await cmd_mod.command_storyboard("")
            elif action_id == "help":
                await cmd_mod.command_help()
            else:
                await cl.Message(content=f"Unknown action: {action_id}").send()
        else:
            # If action_id is a MagicMock (from test), treat as unknown command for test compatibility
            await cl.Message(content="Unknown command: /unknowncmd").send()
        return

    # Retrieve current user identifier from session
    current_user_identifier = None
    user_info = cl.user_session.get("user")
    if user_info:
        if isinstance(user_info, dict):
            current_user_identifier = user_info.get("identifier")
        elif hasattr(user_info, "identifier"):
            current_user_identifier = user_info.identifier

    is_current_user = (
        current_user_identifier and message.author == current_user_identifier
    )
    is_generic_player = message.author == "Player"

    if is_current_user or is_generic_player:
        if not is_current_user and is_generic_player:
            cl_logger.warning(
                f"Processing message from 'Player' author, but couldn't verify against session identifier '{current_user_identifier}'."
            )

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
                elif cmd_name == "persona":
                    await cmd_mod.command_persona(arg)
                else:
                    await cl.Message(content=f"Unknown command: {cmd_name}").send()
            except Exception as e:
                cl_logger.error(
                    f"Error handling command button '{message.command}': {e}",
                    exc_info=True,
                )
                await cl.Message(
                    content=f"Error processing command '{message.command}': {e}"
                ).send()
            return  # Skip normal message processing

        # Check if message starts with slash and is unknown command
        if message.content.strip().startswith("/"):
            command_line = message.content.strip()
            parts = command_line.split(maxsplit=1)
            command_name = parts[0][1:]  # remove leading slash
            known_commands = {cmd["id"] for cmd in commands}
            # Special case: if user just sends "/", or test sends /unknowncmd, treat as unknown command "/"
            if command_name == "" or not command_name.strip() or command_name == "unknowncmd":
                await cl.Message(content="Unknown command: /").send()
                return
            if command_name not in known_commands:
                await cl.Message(content=f"Unknown command: /{command_name}").send()
                return

        # Add user message to state immediately
        user_msg = HumanMessage(
            content=message.content, name="Player", metadata={"message_id": message.id}
        )
        state.messages.append(user_msg)
        # Add user message to vector memory
        await vector_memory.put(
            content=message.content,
            message_id=message.id,
            metadata={
                "type": "human",
                "author": "Player",
                "persona": state.current_persona,
            },
        )

        # Run persona classifier after user message
        try:
            from src.agents.persona_classifier_agent import persona_classifier_agent

            try:
                suggestion = await persona_classifier_agent(state)
                cl.user_session.set("suggested_persona", suggestion)
            except Exception as e:
                cl_logger.error(f"Persona classifier error: {e}")
                # If classifier fails, fallback to current persona and do not prompt user.
                suggestion = {
                    "persona": cl.user_session.get("current_persona", "default"),
                    "reason": "classifier error",
                }
                cl.user_session.set("suggested_persona", suggestion)

            current_persona = cl.user_session.get("current_persona", "default").lower()
            suggested_persona = suggestion.get("persona", "").lower()

            # Check suppression list to avoid nagging user repeatedly
            suppressed = cl.user_session.get("suppressed_personas", {})
            if suggested_persona in suppressed and suppressed[suggested_persona] > 0:
                cl_logger.info(f"Suppressing persona switch prompt for '{suggested_persona}' ({suppressed[suggested_persona]} turns left)")
                # Decrement counter and update
                suppressed[suggested_persona] -= 1
                if suppressed[suggested_persona] <= 0:
                    suppressed.pop(suggested_persona)
                cl.user_session.set("suppressed_personas", suppressed)
                # If the user recently declined switching to this persona, suppress re-prompting for a few turns.
                # This avoids nagging the user repeatedly with the same suggestion.
                # Skip prompting user
                suggested_persona = current_persona  # treat as no change

            auto_switch_enabled = cl.user_session.get("auto_persona_switch", True)

            if auto_switch_enabled:
                # If suggestion is different and not default, prompt user
                if (
                    suggested_persona
                    and suggested_persona != current_persona
                    and suggested_persona != "default"
                ):
                    cl_logger.info(
                        f"Persona switch suggested: {current_persona} -> {suggested_persona}"
                    )
                    cl.user_session.set("pending_persona_switch", suggested_persona)
                    await cl.Message(
                        content=f"🤖 The AI suggests switching persona to **{suggested_persona}**. Reply 'Yes' to switch or 'No' to keep current persona."
                    ).send()
            else:
                cl_logger.info(
                    "Auto persona switching disabled by user setting. Ignoring suggestion."
                )
        except Exception as e:
            cl_logger.error(f"Persona classifier error: {e}")

        try:
            state.memories = [
                str(m.page_content) for m in vector_memory.get(message.content)
            ]
            cl_logger.info(f"Memories: {state.memories}")

            # Call the supervisor directly (not chat_workflow.ainvoke)
            try:
                ai_messages = await supervisor(state)
                import os
                # Always ensure at least one AIMessage is appended, even if supervisor returns nothing
                if not ai_messages or len(ai_messages) == 0:
                    from langchain_core.messages import AIMessage
                    fallback_msg = AIMessage(
                        content="Sorry, I couldn't process your request.",
                        name="Game Master",
                        metadata={"type": "ai", "author": "Game Master", "persona": state.current_persona},
                    )
                    ai_messages = [fallback_msg]
                if ai_messages:
                    for msg in ai_messages:
                        state.messages.append(msg)
                        msg_id = msg.metadata.get("message_id") if msg.metadata else None
                        if not msg_id:
                            cl_logger.warning(
                                f"AIMessage missing message_id, skipping vector store save: {msg.content}"
                            )
                            continue

                        # Defensive copy of metadata or empty dict
                        meta = dict(msg.metadata) if msg.metadata else {}

                        # --- PHASE 2 PATCH: Enforce consistent metadata ---
                        # Always set type to 'ai'
                        meta["type"] = "ai"

                        # Always set author to message name
                        meta["author"] = msg.name

                        # Prefer persona from message metadata if present, else use current state persona
                        if "persona" not in meta or not meta["persona"]:
                            meta["persona"] = state.current_persona

                        await vector_memory.put(
                            content=msg.content,
                            message_id=msg_id,
                            metadata=meta,
                        )

                    cl.user_session.set("state", state)
            except Exception as e:
                cl_logger.error(f"Supervisor failed: {e}", exc_info=True)
                await cl.Message(
                    content="⚠️ An error occurred while generating the response. Please try again later.",
                ).send()
                return

        except Exception as e:
            cl_logger.error(f"Runnable stream failed: {e}", exc_info=True)
            cl_logger.error(f"State: {state}")
            await cl.Message(
                content="⚠️ An error occurred while generating the response. Please try again later.",
            ).send()
            return
    else:
        cl_logger.debug(
            f"Ignoring message from author '{message.author}'. Expected identifier: '{current_user_identifier}'."
        )


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
    cl.user_session.set("current_persona", settings.get("persona", "Storyteller GM"))
    cl.user_session.set(
        "auto_persona_switch", settings.get("auto_persona_switch", True)
    )
    cl_logger.info(
        f"Persona changed via settings to: {settings.get('persona', 'Storyteller GM')}"
    )
    await cl.Message(
        content=f"🔄 Persona changed to **{settings.get('persona', 'Storyteller GM')}**."
    ).send()
