import os
import asyncio

from chainlit import on_chat_start, on_chat_resume, on_message
from chainlit.types import ThreadDict

from memory_management import get_chat_memory, get_vector_memory
from state_graph import call_storyboard_generation, graph, call_image_generation

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.tool import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import chainlit as cl
from chainlit import Message as CLMessage
from chainlit.input_widget import Select

from config import (
    CHAINLIT_STARTERS,
    KNOWLEDGE_DIRECTORY,
    NUM_IMAGE_PROMPTS,
)

@on_chat_start
async def on_chat_start():
    """
    Event handler for when a new chat starts. Initializes chat and vector memories, image generation memory,
    sets up the runnable, loads knowledge documents, and sends starter messages.
    """

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=0,
            )
        ]
    ).send()
    value = settings["Model"]

    # Initialize memories
    cl.user_session.set("chat_memory", get_chat_memory())
    cl.user_session.set("vector_memory", get_vector_memory())
    cl.user_session.set("image_generation_memory", [])  # Initialize image generation memory
    cl.user_session.set("ai_message_id", None)

    # Setup runnable
    cl.user_session.set("runnable", graph)

    # Load knowledge documents into vector store
    await load_knowledge_documents()

    # Send Chainlit starters
    for starter in CHAINLIT_STARTERS:
        await CLMessage(content=starter).send()
        # Track in history
        chat_memory = cl.user_session.get("chat_memory")
        chat_memory.chat_memory.add_ai_message(starter)
        vector_memory = cl.user_session.get("vector_memory")
        vector_memory.save_context({"AI": starter}, {"AI": starter})

@on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """
    Event handler for resuming a chat. Reconstructs chat, vector, and image generation memories from the thread history.

    Args:
        thread (ThreadDict): The thread dictionary containing chat steps.
    """
    # Set the user in the session
    user_dict = thread.get('user')
    if user_dict:
        cl.user_session.set('user', cl.User(**user_dict))

    cl.user_session.set("ai_message_id", None)

    chat_memory = get_chat_memory()
    vector_memory = get_vector_memory()
    image_generation_memory = []

    # Sort messages by createdAt to ensure correct order
    sorted_steps = sorted(thread.get("steps", []), key=lambda m: m.get("createdAt", ""))

    message_pair = []
    for message in sorted_steps:
        message_prefix = None
        if message["type"] == "user_message":
            chat_memory.chat_memory.add_user_message(message["output"])
            message_prefix = "HUMAN"
        elif message["type"] == "ai_message":
            chat_memory.chat_memory.add_ai_message(message["output"])
            message_prefix = "AI"
        elif message["type"] == "image_generation":
            image_generation_memory.append(message["output"])
            continue
        else:
            continue

        message_pair.append((message_prefix, message["output"]))

        if len(message_pair) == 2:
            message_in = message_pair[0]
            message_out = message_pair[1]
            vector_memory.save_context({message_in[0]: message_in[1]}, {message_out[0]: message_out[1]})
            message_pair = [message_pair[1]]

    cl.user_session.set("chat_memory", chat_memory)
    cl.user_session.set("vector_memory", vector_memory)
    cl.user_session.set("image_generation_memory", image_generation_memory)

    # Setup runnable
    cl.user_session.set("runnable", graph)

    # Load knowledge documents into vector store
    await load_knowledge_documents()

@on_message
async def on_message(message: CLMessage):
    """
    Event handler for incoming messages. Processes user input, generates AI response, initiates image generation,
    and updates memories.

    Args:
        message (CLMessage): The incoming message from the user.
    """
    chat_memory = cl.user_session.get("chat_memory")  # type: ConversationSummaryBufferMemory
    vector_memory = cl.user_session.get("vector_memory")  # type: VectorStoreRetrieverMemory
    image_generation_memory = cl.user_session.get("image_generation_memory", [])  # type: list
    runnable = cl.user_session.get("runnable")  # type: Runnable

    if message.type != "user_message":
        return

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()

    try:
        # Generate AI response
        ai_response = CLMessage(content="")
        history = chat_memory.chat_memory.messages + [HumanMessage(content=message.content)] 
        async for msg, metadata in runnable.astream(input={"messages": history}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
            if (
                msg.content
                and not isinstance(msg, HumanMessage)
                and metadata["langgraph_node"] == "writer"
            ):
                await ai_response.stream_token(msg.content)
        await ai_response.send()
        history += [ai_response]

        # After sending the AI response, get its message ID
        ai_message_id = ai_response.id
        cl.user_session.set("ai_message_id", ai_message_id) # Assuming CLMessage.send() populates the 'id' attribute
        cl.logger.debug(f"AI Response Message ID: {ai_message_id}")

        asyncio.create_task(call_image_generation(history, ai_message_id))
    except Exception as e:
        cl.logger.error(f"Runnable stream failed: {e}")
        cl.logger.error(f"Metadata: {metadata}")  # Log the metadata to see if 'action' is present
        await CLMessage(content="⚠️ An error occurred while generating the response. Please try again later.").send()
        return

    # Update memories with the latest interaction
    chat_memory.chat_memory.add_user_message(message.content)
    chat_memory.chat_memory.add_ai_message(ai_response.content)
    vector_memory.save_context({"HUMAN": message.content}, {"AI": ai_response.content})

async def load_knowledge_documents():
    """
    Loads documents from the knowledge directory into the vector store.
    """
    if not os.path.exists(KNOWLEDGE_DIRECTORY):
        cl.logger.warning(f"Knowledge directory '{KNOWLEDGE_DIRECTORY}' does not exist. Skipping document loading.")
        return

    vector_memory = cl.user_session.get("vector_memory", None)

    if not vector_memory:
        cl.logger.error("Vector memory not initialized.")
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
                cl.logger.warning(f"Unsupported file type: {file}. Skipping.")
                continue
            try:
                loaded_docs = loader.load()
                split_docs = text_splitter.split_documents(loaded_docs)
                documents.extend(split_docs)
            except Exception as e:
                cl.logger.error(f"Error loading document {file_path}: {e}")

    if documents:
        cl.logger.info(f"Adding {len(documents)} documents to the vector store.")
        vector_memory.retriever.vectorstore.add_documents(documents)
    else:
        cl.logger.info("No documents found to add to the vector store.")
