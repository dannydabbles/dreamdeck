import os
import logging
import asyncio
import random
import base64
import httpx
from typing import List, Optional
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
from src.agents.dice_agent import dice_roll_agent
from src.agents.web_search_agent import web_search_agent
from chainlit import user_session as cl_user_session  # Import cl_user_session
from langchain_core.callbacks.manager import CallbackManagerForChainRun  # Import CallbackManagerForChainRun

from langchain_core.stores import BaseStore

import chainlit as cl
from chainlit.input_widget import Slider, TextInput  # Import widgets
from src import config  # Import your config

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

    Sets up the user session, initializes the chat state, and sends initial messages.
    """
    # Import commands *inside* the function to ensure Chainlit is initialized
    from src import commands

    try:
        # Initialize vector store
        vector_memory = VectorStore()
        cl.user_session.set("vector_memory", vector_memory)

        # Initialize agents
        cl_user_session.set("decision_agent", decision_agent)
        cl_user_session.set("writer_agent", writer_agent)
        cl_user_session.set("storyboard_editor_agent", storyboard_editor_agent)
        cl_user_session.set("dice_roll_agent", dice_roll_agent)
        cl_user_session.set("web_search_agent", web_search_agent)

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
                Slider(
                    id="storyboard_temp",
                    label="Storyboard Agent - Temperature",
                    min=0.0, max=2.0, step=0.1, initial=config.STORYBOARD_EDITOR_AGENT_TEMPERATURE
                ),
                Slider(
                    id="storyboard_max_tokens",
                    label="Storyboard Agent - Max Tokens",
                    min=100, max=16000, step=100, initial=config.STORYBOARD_EDITOR_AGENT_MAX_TOKENS
                ),
                TextInput(
                    id="storyboard_endpoint",
                    label="Storyboard Agent - OpenAI Endpoint URL",
                    initial=config.STORYBOARD_EDITOR_AGENT_BASE_URL or "",
                    placeholder="e.g., http://localhost:5000/v1"
                ),
                Slider(
                    id="decision_temp",
                    label="Decision Agent - Temperature",
                    min=0.0, max=2.0, step=0.1, initial=config.DECISION_AGENT_TEMPERATURE
                ),
                Slider(
                    id="decision_max_tokens",
                    label="Decision Agent - Max Tokens",
                    min=10, max=1000, step=10, initial=config.DECISION_AGENT_MAX_TOKENS
                ),
                TextInput(
                    id="decision_endpoint",
                    label="Decision Agent - OpenAI Endpoint URL",
                    initial=config.DECISION_AGENT_BASE_URL or "",
                    placeholder="e.g., http://localhost:5000/v1"
                ),
            ]
        ).send()

        # Launch knowledge loading in the background
        asyncio.create_task(load_knowledge_documents())

        # Initialize thread in Chainlit with a start message
        await cl.Message(content=START_MESSAGE, author="Game Master").send()

        # Create initial state
        state = ChatState(
            messages=[AIMessage(content=START_MESSAGE, name="Game Master")],
            thread_id=cl.context.session.thread_id,
            user_preferences=cl.user_session.get("user_session", {}).get(
                "preferences", {}
            ),
        )

        # Store state
        cl.user_session.set("state", state)
        cl.user_session.set("image_generation_memory", [])
        cl.user_session.set("ai_message_id", None)

    except Exception as e:
        cl_logger.error(f"Application failed to start: {e}", exc_info=True)
        raise
    finally:
        # Close database pool
        await DatabasePool.close()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread.

    Args:
        thread (ThreadDict): The thread dictionary from Chainlit.
    """
    # Set the user in the session
    user_dict = thread.get("user")
    if user_dict:
        cl.user_session.set("user", user_dict)

    # Initialize vector store
    vector_memory = VectorStore()
    cl.user_session.set("vector_memory", vector_memory)

    # Initialize thread in Chainlit with a start message
    messages = []
    image_generation_memory = []
    cl.user_session.set("gm_message", cl.Message(content="", author="Game Master"))

    # Reconstruct messages from thread history
    for step in sorted(thread.get("steps", []), key=lambda m: m.get("createdAt", "")):
        if step["type"] == "user_message":
            messages.append(HumanMessage(content=step["output"], name="Player"))
            vector_memory.put(content=step["output"])
        elif step["type"] == "assistant_message":
            messages.append(AIMessage(content=step["output"], name=step["name"]))
            vector_memory.put(content=step["output"])

    # Create state
    state = ChatState(
        messages=messages,
        thread_id=thread["id"],
        user_preferences=cl.user_session.get("user_session", {}).get("preferences", {}),
        thread_data=cl.user_session.get("thread_data", {}),
    )

    # Store state and memories
    cl.user_session.set("state", state)
    cl.user_session.set("image_generation_memory", image_generation_memory)
    cl.user_session.set("ai_message_id", None)

    # Load knowledge documents
    await load_knowledge_documents()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages.

    Args:
        message (cl.Message): The incoming message.
    """
    state: ChatState = cl.user_session.get("state")
    vector_memory: VectorStore = cl.user_session.get("vector_memory")

    if message.type != "user_message":
        return

    try:
        # Add user message to state
        user_msg = HumanMessage(content=message.content, name="Player", metadata={"message_id": message.id})
        state.messages.append(user_msg)

        # Add user message to vector memory
        await vector_memory.put(content=message.content, message_id=message.id, metadata={"type": "human", "author": "Player"})

        # Put messages relevant to the player message into state.memories list for AI to use from the vector memory
        state.memories = [
            str(m.page_content) for m in vector_memory.get(message.content)
        ]
        cl_logger.info(f"Memories: {state.memories}")

        # Generate AI response using the chat workflow
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
        # state = await chat_workflow.ainvoke(input={"messages": state.messages, "store": cl.user_session.get("vector_memory"), "previous": state}, config=thread_config)
        # gm_message = cl.Message(content="")

        inputs = {"messages": state.messages, "previous": state}
        state = await chat_workflow.ainvoke(
            inputs, config=RunnableConfig(callbacks=[cb], **thread_config)
        )

        # Store final AI response in vector store
        if state.messages and isinstance(state.messages[-1], AIMessage):
            ai_msg = state.messages[-1]
            if ai_msg.metadata and "message_id" in ai_msg.metadata:
                await vector_memory.put(content=ai_msg.content, message_id=ai_msg.metadata["message_id"], metadata={"type": "ai", "author": ai_msg.name})
            else:
                cl_logger.warning(f"Final AIMessage missing message_id: {ai_msg.content}")

        cl.user_session.set("state", state)

    except Exception as e:
        cl.element.logger.error(f"Runnable stream failed: {e}", exc_info=True)
        cl.element.logger.error(f"State: {state}")  # Log the state
        await cl.Message(
            content="⚠️ An error occurred while generating the response. Please try again later.",
        ).send()
        return


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
                split_docs = await loop.run_in_executor(
                    None, text_splitter.split_documents, loaded_docs
                )
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
