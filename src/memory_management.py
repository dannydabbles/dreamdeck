import os
import uuid

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from langchain.memory import ConversationSummaryBufferMemory, VectorStoreRetrieverMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import (
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_STREAMING,
    LLM_TIMEOUT,
    LLM_PRESENCE_PENALTY,
    LLM_FREQUENCY_PENALTY,
    LLM_TOP_P,
    LLM_VERBOSE,
)

from langchain_openai import ChatOpenAI

import logging

# Initialize logging
cl_logger = logging.getLogger("chainlit")

def get_chat_memory() -> ConversationSummaryBufferMemory:
    """
    Initializes the conversation summary buffer memory using ChatOpenAI.

    Returns:
        ConversationSummaryBufferMemory: The initialized chat memory.
    """
    cl_logger.info("Initializing chat memory.")
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(
            base_url="http://192.168.1.111:5000/v1",
            temperature=LLM_TEMPERATURE,
            streaming=LLM_STREAMING,
            model_name=LLM_MODEL_NAME,
            request_timeout=LLM_TIMEOUT,
            max_tokens=LLM_MAX_TOKENS,
            presence_penalty=LLM_PRESENCE_PENALTY,
            frequency_penalty=LLM_FREQUENCY_PENALTY,
            top_p=LLM_TOP_P,
            verbose=LLM_VERBOSE
        ),
        max_token_limit=40000,
        return_messages=True
    )
    cl_logger.info("Chat memory initialized.")
    return memory

def get_vector_memory() -> VectorStoreRetrieverMemory:
    """
    Initializes the vector memory using HuggingFace Embeddings and Chroma vector store.

    Returns:
        VectorStoreRetrieverMemory: The initialized vector memory.
    """
    cl_logger.info("Initializing vector memory.")
    # This text splitter is used to create the parent documents
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # This text splitter is used to create the child documents
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    # The vectorstore to use to index the child chunks
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vectorstore = Chroma(
        collection_name=uuid.uuid4().hex,
        embedding_function=embeddings,
        persist_directory=os.path.join(os.getcwd(), "chroma_db")
    )

    retriever = vectorstore.as_retriever()
    memory = VectorStoreRetrieverMemory(
        retriever=retriever,
    )
    cl_logger.info("Vector memory initialized.")
    return memory
