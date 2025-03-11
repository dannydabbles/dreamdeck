from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from chromadb import PersistentClient
from typing import Dict, Any, List, Sequence, Tuple, Optional, Iterator  # Import Optional and Iterator
import os
import chainlit as cl
import asyncio
from src.config import parse_size, CACHING_SETTINGS  # Import parse_size and CACHING_SETTINGS

class VectorStore(BaseStore):
    """Custom vector store implementation using ChromaDB for persistent storage.

    Attributes:
        embeddings (HuggingFaceEmbeddings): The embeddings model.
        client (PersistentClient): The ChromaDB client.
        collection: The ChromaDB collection.
    """

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.client = PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="my_collection",
            embedding_function=self.embeddings
        )

    # Required abstract methods from BaseStore
    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Batch-get documents by IDs."""
        return [self.vectorstore.get(id=key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Batch-set documents."""
        for key, doc in key_value_pairs:
            self.put(doc.page_content)  # Delegate to existing 'put' method

    def mdelete(self, keys: Sequence[str]) -> None:
        """Batch-delete documents by IDs."""
        self.vectorstore.delete(ids=list(keys))

    def yield_keys(self, prefix: str = None) -> Iterator[str]:
        """Iterate over keys matching a prefix."""
        for doc in self.vectorstore.get():
            if prefix is None or doc.id.startswith(prefix):
                yield doc.id

    def get(self, field: str) -> List[Document]:
        """Get relevant documents using ChromaDB.

        Args:
            field (str): The field to search.

        Returns:
            List[Document]: The relevant documents.
        """
        try:
            # Get relevant documents using embeddings
            try:
                query_embedding = self.embeddings.embed_query(field)
                docs = self.vectorstore.similarity_search(query_embedding, k=3)
                return [
                    Document(page_content=d.page_content, metadata=d.metadata)
                    for d in docs
                ]
            except Exception as e:
                cl.logger.error(f"Embedding or search error: {e}")
                return []

        except Exception as e:
            cl.logger.error(f"Error in vector store get operation: {e}")
            return []

    def put(self, content: str) -> None:
        """Store new content in ChromaDB."""
        doc = Document(page_content=content, metadata={"source": "user_input"})
        self.vectorstore.add_documents([doc])
        self.vectorstore.persist()

    async def batch(self, operations: Sequence[Tuple[str, tuple, str, Any]]) -> None:
        """Execute multiple operations in batch asynchronously.

        Args:
            operations (Sequence[Tuple[str, tuple, str, Any]]): The operations to execute.
        """
        await asyncio.gather(
            *[
                asyncio.create_task(self._async_operation(op, key, field, value))
                for op, key, field, value in operations
            ]
        )

    async def _async_operation(
        self, op: str, key: tuple, field: str, value: Any
    ) -> None:
        """Helper method for async batch operations.

        Args:
            op (str): The operation type.
            key (tuple): The key for the document.
            field (str): The field to operate on.
            value (Any): The value to store.
        """
        if op == "get":
            self.get(key, field)
        elif op == "put":
            self.put(key, field, value)

# Ensure max_size is parsed correctly
max_cache_size = parse_size(CACHING_SETTINGS.get('max_size', '100MB'))
