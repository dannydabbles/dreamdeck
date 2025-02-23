import os
import asyncio

from chainlit import on_chat_start, on_chat_resume, on_message
from chainlit.types import ThreadDict

from memory_management import get_chat_memory, get_vector_memory
from state_graph import story_workflow as graph, generate_storyboard
from image_generation import handle_image_generation, generate_image_generation_prompts
from models import ChatState
from config import AI_WRITER_PROMPT, CHAINLIT_STARTERS

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
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
    """Initialize new chat session with Chainlit integration."""
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
    
    # Create initial state
    state = ChatState(
        messages=[SystemMessage(content=AI_WRITER_PROMPT)],
        thread_id=cl.context.session.id
    )
    
    # Initialize thread in Chainlit
    await cl.Message(content=AI_WRITER_PROMPT, author="system").send()
    
    # Store state
    cl.user_session.set("state", state)
    cl.user_session.set("image_generation_memory", [])
    cl.user_session.set("ai_message_id", None)
    
    # Setup runnable
    cl.user_session.set("runnable", graph)
    
    # Load knowledge documents
    await load_knowledge_documents()
    
    # Send starters
    for starter in CHAINLIT_STARTERS:
        msg = await cl.Message(content=starter).send()
        state.messages.append(AIMessage(content=starter, additional_kwargs={"message_id": msg.id}))

@on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread."""
    # Set the user in the session
    user_dict = thread.get('user')
    if user_dict:
        cl.user_session.set('user', cl.User(**user_dict))

    messages = [SystemMessage(content=AI_WRITER_PROMPT)]
    image_generation_memory = []
    
    # Reconstruct messages from thread history
    for step in sorted(thread.get("steps", []), key=lambda m: m.get("createdAt", "")):
        if step["type"] == "user_message":
            messages.append(HumanMessage(
                content=step["output"],
                additional_kwargs={"message_id": step["id"]}
            ))
        elif step["type"] == "ai_message":
            messages.append(AIMessage(
                content=step["output"],
                additional_kwargs={"message_id": step["id"]}
            ))
        elif step["type"] == "image_generation":
            image_generation_memory.append(step["output"])
    
    # Create state
    state = ChatState(
        messages=messages,
        thread_id=thread["id"]
    )
    
    # Store state and memories
    cl.user_session.set("state", state)
    cl.user_session.set("image_generation_memory", image_generation_memory)
    cl.user_session.set("ai_message_id", None)
    
    # Setup runnable
    cl.user_session.set("runnable", graph)
    
    # Load knowledge documents
    await load_knowledge_documents()

@on_message
async def on_message(message: CLMessage):
    """Handle incoming messages."""
    state = cl.user_session.get("state")
    runnable = cl.user_session.get("runnable")

    if message.type != "user_message":
        return

    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()

    try:
        # Add user message to state
        state = cl.user_session.get("state")
        state.messages.append(HumanMessage(content=message.content))
        
        # Generate AI response
        ai_response = CLMessage(content="")
        async for chunk in runnable.astream(
            state,
            config=RunnableConfig(callbacks=[cb], **config)
        ):
            if isinstance(chunk, dict) and chunk.get("messages"):
                await ai_response.stream_token(chunk["messages"][-1].content)
        
        await ai_response.send()
        
        # Update state with the new message ID
        ai_message_id = ai_response.id
        state.metadata["current_message_id"] = ai_message_id
        
        # Handle image generation if there's a storyboard
        if "storyboard" in chunk:
            asyncio.create_task(handle_image_generation(
                await generate_image_generation_prompts(chunk["storyboard"]),
                ai_message_id
            ))
        
        # Update session state
        cl.user_session.set("state", state)
    except Exception as e:
        cl.logger.error(f"Runnable stream failed: {e}")
        cl.logger.error(f"State metadata: {state.metadata}")  # Log the state's metadata
        await CLMessage(content="⚠️ An error occurred while generating the response. Please try again later.").send()
        return

    # Update session state
    cl.user_session.set("state", state)

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
