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

# Define Chainlit commands for UI buttons and slash menu
from chainlit import Action

# List of available commands/actions for UI (used for both slash and action buttons)
COMMANDS = [
    {"id": "roll", "icon": "dice-5", "description": "Roll dice"},
    {"id": "search", "icon": "globe", "description": "Web search"},
    {"id": "todo", "icon": "list", "description": "Add a TODO"},
    {"id": "write", "icon": "pen-line", "description": "Direct prompt to writer"},
    {"id": "storyboard", "icon": "image", "description": "Generate storyboard"},
    {"id": "report", "icon": "bar-chart", "description": "Generate daily report"},
    {"id": "persona", "icon": "user", "description": "Switch persona"},
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
    """
    Initialize new chat session with Chainlit integration.

    Sets up the user session, initializes the chat state, initializes agents and vector store,
    and sends initial messages. Agents and vector store are stored in the Chainlit user session.
    """
    # Chainlit v2+: Use ChatSettings and Action objects for UI. No cl.command, cl.set_chat_commands, or cl.set_commands.

    # Import load_knowledge_documents so it is defined in this scope
    global load_knowledge_documents

    # Determine active persona from profile selection, default if none
    persona = cl.user_session.get("current_persona", "Friend")

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

        # Define Chat Settings with persona selector and LLM options
        chat_settings = cl.ChatSettings(
            [
                Select(
                    id="persona",
                    label="Persona",
                    values=[
                        "Storyteller GM",
                        "Friend",
                        "Therapist",
                        "Secretary",
                        "Coder",
                        "Dungeon Master",
                        "Default",
                    ],
                    initial="Friend",  # Changed default persona for new chats
                ),
                Slider(
                    id="llm_temperature",
                    label="LLM Temperature",
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    initial=config.LLM_TEMPERATURE,
                ),
                Slider(
                    id="llm_max_tokens",
                    label="LLM Max Tokens",
                    min=100,
                    max=16000,
                    step=100,
                    initial=config.LLM_MAX_TOKENS,
                ),
                TextInput(
                    id="llm_endpoint",
                    label="LLM Endpoint URL",
                    initial=config.OPENAI_SETTINGS.get("base_url", "") or "",
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
        )
        settings = await chat_settings.send()

        # Defensive: if settings is None (e.g., in tests), replace with empty dict
        if settings is None:
            settings = {}

        cl.user_session.set("chat_settings", settings)
        cl.user_session.set("current_persona", settings.get("persona", "Friend"))
        cl.user_session.set(
            "auto_persona_switch", settings.get("auto_persona_switch", True)
        )
        cl.user_session.set(
            "llm_temperature", settings.get("llm_temperature", config.LLM_TEMPERATURE)
        )
        cl.user_session.set(
            "llm_max_tokens", settings.get("llm_max_tokens", config.LLM_MAX_TOKENS)
        )
        cl.user_session.set(
            "llm_endpoint",
            settings.get("llm_endpoint", config.OPENAI_SETTINGS.get("base_url", "")),
        )

        # Launch knowledge loading in the background
        asyncio.create_task(load_knowledge_documents())

        # Initialize thread in Chainlit with a start message
        # Use Chainlit Action objects for UI buttons (v1.0+)
        actions = [
            Action(
                id=cmd["id"], name=cmd["description"], icon=cmd.get("icon"), payload={}
            )
            for cmd in COMMANDS
        ]
        start_cl_msg = cl.Message(
            content=START_MESSAGE,
            author=current_user_identifier,
            actions=actions,
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

        # Register commands for Chainlit UI (slash menu, etc.)
        await cl.context.emitter.set_commands(COMMANDS)

    except Exception as e:
        cl_logger.error(f"Application failed to start: {e}", exc_info=True)
        raise


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread.

    Initializes agents and vector store, reconstructs chat state from thread history,
    and stores them in the Chainlit user session.
    """

    # Import load_knowledge_documents so it is defined in this scope
    global load_knowledge_documents

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


# Register Chainlit action callbacks for UI buttons using the modern API
from chainlit import Action


@cl.action_callback("roll")
async def on_roll_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_roll("")


@cl.action_callback("search")
async def on_search_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_search("")


@cl.action_callback("todo")
async def on_todo_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_todo("")


@cl.action_callback("write")
async def on_write_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_write("")


@cl.action_callback("storyboard")
async def on_storyboard_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_storyboard("")


@cl.action_callback("report")
async def on_report_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_report()


@cl.action_callback("help")
async def on_help_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_help()


@cl.action_callback("reset")
async def on_reset_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_reset()


@cl.action_callback("save")
async def on_save_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_save()


@cl.action_callback("persona")
async def on_persona_action(action: Action):
    from src import commands as cmd_mod

    await cmd_mod.command_persona("")


@cl.on_message
async def on_message(message: cl.Message):
    """
    Handle incoming user chat messages.

    - If the message is a slash command (e.g. "/roll 2d6"), dispatch to the corresponding handler.
    - Otherwise, treat as a normal speech turn and call the supervisor to decide which agent/tool/persona to use.
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

            append_log(
                pending_persona,
                f"Persona switched to {pending_persona} by user confirmation.",
            )
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

        # --- SLASH COMMAND HANDLING ---
        if message.content.strip().startswith("/"):
            command_line = message.content.strip()
            # Special case: if user just sends "/", treat as unknown command "/"
            if command_line == "/" or command_line.strip() == "/":
                await cl.Message(content="Unknown command: /").send()
                return
            # Extract command name and argument
            parts = command_line[1:].split(maxsplit=1)
            command_name = parts[0].strip()
            arg = parts[1] if len(parts) > 1 else ""
            known_commands = {cmd["id"] for cmd in COMMANDS}
            if not command_name:
                await cl.Message(content="Unknown command: /").send()
                return
            if command_name not in known_commands:
                await cl.Message(content=f"Unknown command: /{command_name}").send()
                return
            # Dispatch to the corresponding command handler
            try:
                from src import commands as cmd_mod

                if command_name == "roll":
                    await cmd_mod.command_roll(arg)
                elif command_name == "search":
                    await cmd_mod.command_search(arg)
                elif command_name == "todo":
                    await cmd_mod.command_todo(arg)
                elif command_name == "write":
                    await cmd_mod.command_write(arg)
                elif command_name == "storyboard":
                    await cmd_mod.command_storyboard(arg)
                elif command_name == "report":
                    await cmd_mod.command_report()
                elif command_name == "help":
                    await cmd_mod.command_help()
                elif command_name == "reset":
                    await cmd_mod.command_reset()
                elif command_name == "save":
                    await cmd_mod.command_save()
                elif command_name == "persona":
                    await cmd_mod.command_persona(arg)
                else:
                    await cl.Message(content=f"Unknown command: /{command_name}").send()
            except Exception as e:
                cl_logger.error(
                    f"Error handling slash command '/{command_name}': {e}",
                    exc_info=True,
                )
                await cl.Message(
                    content=f"Error processing command '/{command_name}': {e}"
                ).send()
            return  # Skip normal message processing

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

        # After user message, always call the supervisor to decide next step
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
                        metadata={
                            "type": "ai",
                            "author": "Game Master",
                            "persona": state.current_persona,
                        },
                    )
                    ai_messages = [fallback_msg]
                if ai_messages:
                    for msg in ai_messages:
                        state.messages.append(msg)
                        metadata = getattr(msg, "metadata", None)
                        msg_id = metadata.get("message_id") if metadata else None
                        if not msg_id:
                            # Attempt to generate a synthetic message_id to avoid skipping vector store save
                            import uuid

                            msg_id = str(uuid.uuid4())
                            cl_logger.warning(
                                f"AIMessage missing message_id, generated synthetic ID: {msg_id} for content: {msg.content}"
                            )

                        # Defensive copy of metadata or empty dict
                        meta = dict(metadata) if metadata else {}

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


# --- load_knowledge_documents definition moved up for visibility ---
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
    cl.user_session.set(
        "llm_temperature", settings.get("llm_temperature", config.LLM_TEMPERATURE)
    )
    cl.user_session.set(
        "llm_max_tokens", settings.get("llm_max_tokens", config.LLM_MAX_TOKENS)
    )
    cl.user_session.set(
        "llm_endpoint",
        settings.get("llm_endpoint", config.OPENAI_SETTINGS.get("base_url", "")),
    )
    cl_logger.info(
        f"Persona changed via settings to: {settings.get('persona', 'Storyteller GM')}"
    )
    await cl.Message(
        content=f"🔄 Persona changed to **{settings.get('persona', 'Storyteller GM')}**."
    ).send()
