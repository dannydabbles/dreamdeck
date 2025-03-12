import os
import logging
import asyncio
import random
import base64
import httpx
from typing import List, Optional, Dict as ThreadDict
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, FunctionMessage, ToolMessage
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableConfig  # Import RunnableConfig
from config import (
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
    START_MESSAGE
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

import chainlit as cl

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
    else:
        return None

@cl.on_chat_start
async def on_chat_start():
    """Initialize new chat session with Chainlit integration.

    Sets up the user session, initializes the chat state, and sends initial messages.
    """
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
        
        # Initialize thread in Chainlit with a start message
        await cl.Message(content=START_MESSAGE, author="Game Master").send()

        # Create initial state
        state = ChatState(
            messages=[AIMessage(content=START_MESSAGE)],
            thread_id=cl.context.session.thread_id,
            user_preferences=cl.user_session.get("user_session", {}).get("preferences", {})
        )

        # Store state
        cl.user_session.set("state", state)
        cl.user_session.set("image_generation_memory", [])
        cl.user_session.set("ai_message_id", None)

        # Initialize vector store
        cl.user_session.set("vector_memory", VectorStore())

        # Setup runnable
        cl.user_session.set("runnable", chat_workflow)

        # Load knowledge documents
        await load_knowledge_documents()

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
    await cl.Message(content=START_MESSAGE, author="system").send()
    vector_memory.put(content=START_MESSAGE)

    messages = [AIMessage(content=START_MESSAGE, type="system")]
    image_generation_memory = []

    # Reconstruct messages from thread history
    for step in sorted(thread.get("steps", []), key=lambda m: m.get("createdAt", "")):
        if step["type"] == "user_message":
            messages.append(HumanMessage(content=step["output"], name="Player"))
        elif step["type"] == "ai_message":
            messages.append(AIMessage(content=step["output"], name="GM"))
        elif step["type"] == "tool":
            tool_call = step.get("output")
            messages.append(
                ToolMessage(
                    content=step['output'],
                    tool_call_id=step['id']
                )
            )
        elif step["type"] == "image_generation":
            image_generation_memory.append(step["output"])

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

    # Setup runnable
    cl.user_session.set("runnable", chat_workflow)

    # Load knowledge documents
    await load_knowledge_documents()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages.

    Args:
        message (cl.Message): The incoming message.
    """
    state = cl.user_session.get("state")

    if message.type != "user_message":
        return

    try:
        # Add user message to state
        state.messages.append({"content": message.content, "type": "user"})

        # Add user message to vector memory
        vector_memory = cl.user_session.get("vector_memory")
        vector_memory.put(content=message.content)

        # Generate AI response using the chat workflow
        state = await chat_workflow(state.messages, store=cl.user_session.get("vector_memory"), previous=state)
        cl.user_session.set("state", state)

    except Exception as e:
        cl.element.logger.error(f"Runnable stream failed: {e}", exc_info=True)
        cl.element.logger.error(
            f"State metadata: {state.metadata}"
        )  # Log the state's metadata
        await cl.Message(
            content="⚠️ An error occurred while generating the response. Please try again later."
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

    for root, dirs, files in os.walk(KNOWLEDGE_DIRECTORY):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                cl.element.logger.warning(f"Unsupported file type: {file}. Skipping.")
                continue
            try:
                loaded_docs = loader.load()
                split_docs = text_splitter.split_documents(loaded_docs)
                documents.extend(split_docs)
            except Exception as e:
                cl.element.logger.error(f"Error loading document {file_path}: {e}")

    if documents:
        cl.element.logger.info(
            f"Adding {len(documents)} documents to the vector store."
        )
        vector_memory.vectorstore.add_documents(documents)
        # TODO: Should we persist the ChromaDB vector store here?
    else:
        cl.element.logger.info("No documents found to add to the vector store.")
