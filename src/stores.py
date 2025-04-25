import asyncio
import os  # Add os for env var check
import uuid
from typing import (
    Any,
    Dict,
    Iterator,
    List,  # Import Optional and Iterator
    Optional,
    Sequence,
    Tuple,
)

import chainlit as cl
from chromadb import Client, PersistentClient  # Import both clients
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_core.documents import Document

from src.config import (  # Import parse_size and CACHING_SETTINGS
    CACHING_SETTINGS,
    parse_size,
)


class VectorStore:
    """Custom vector store implementation using ChromaDB for persistent storage.

    Attributes:
        embeddings (HuggingFaceEmbeddings): The embeddings model.
        client (PersistentClient or Client): The ChromaDB client.
        collection: The ChromaDB collection.
    """

    def __init__(self):
        self.embeddings = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        # Use in-memory ChromaDB during tests
        if os.environ.get("DREAMDECK_TEST_MODE") == "1":
            self.client = Client()  # In-memory, ephemeral
        else:
            self.client = PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name=cl.context.session.thread_id, embedding_function=self.embeddings
        )

    def get(self, content: str) -> List[Document]:
        """Get relevant documents using ChromaDB.

        Args:
            content (str): The content to search.

        Returns:
            List[Document]: The relevant documents.
        """
        results = self.collection.query(
            query_texts=[content], n_results=5, include=["documents", "metadatas"]
        )
        documents_flat = results.get("documents", [[]])[0]
        metadatas_flat = results.get("metadatas", [[]])[0]
        # Defensive: replace None metadata with empty dict
        return [
            Document(page_content=doc, metadata=meta or {})
            for doc, meta in zip(documents_flat, metadatas_flat)
        ]

    async def put(
        self, content: str, message_id: str, metadata: Optional[dict] = None
    ) -> None:
        """Store new content in ChromaDB."""
        full_metadata = {"message_id": message_id, **(metadata or {})}
        await asyncio.to_thread(
            self.collection.add,
            ids=[message_id],
            documents=[content],
            metadatas=[full_metadata],
        )

    async def add_documents(self, docs: List[Document]) -> None:
        """Store a list of documents in ChromaDB."""
        doc_ids = [str(uuid.uuid4()) for _ in docs]
        metadatas = [{"source": doc.metadata.get("source", "unknown")} for doc in docs]
        await asyncio.to_thread(
            self.collection.add,
            ids=doc_ids,
            documents=[d.page_content for d in docs],
            metadatas=metadatas,
        )


# Ensure max_size is parsed correctly
max_cache_size = parse_size(CACHING_SETTINGS.get("max_size", "100MB"))
