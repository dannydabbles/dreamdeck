from langchain_core.documents import Document
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient
from typing import Dict, Any, List, Sequence, Tuple, Optional, Iterator  # Import Optional and Iterator
import uuid
import chainlit as cl
import asyncio
from src.config import parse_size, CACHING_SETTINGS  # Import parse_size and CACHING_SETTINGS

class VectorStore:
    """Custom vector store implementation using ChromaDB for persistent storage.

    Attributes:
        embeddings (HuggingFaceEmbeddings): The embeddings model.
        client (PersistentClient): The ChromaDB client.
        collection: The ChromaDB collection.
    """

    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=cl.context.session.thread_id,
            embedding_function=self.embeddings
        )

    def get(self, content: str) -> List[Document]:
        """Get relevant documents using ChromaDB.

        Args:
            content (str): The content to search.

        Returns:
            List[Document]: The relevant documents.
        """
        results = self.collection.query(query_texts=[content], n_results=5)
        documents_flat = results.get('documents', [[]])[0]  # Get first query's results
        return [Document(page_content=doc) for doc in documents_flat]

    async def put(self, content: str) -> None:
        """Store new content in ChromaDB."""
        await asyncio.to_thread(
            self.collection.add,
            ids=[str(uuid.uuid4())],
            documents=[content]
        )

    async def add_documents(self, docs: List[Document]) -> None:
        """Store a list of documents in ChromaDB."""
        for doc in docs:
            await self.put(doc.page_content)

# Ensure max_size is parsed correctly
max_cache_size = parse_size(CACHING_SETTINGS.get('max_size', '100MB'))
