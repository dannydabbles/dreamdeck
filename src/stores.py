from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient
from typing import Dict, Any, List, Sequence, Tuple, Optional, Iterator  # Import Optional and Iterator
import uuid
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
        self.embeddings = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="my_collection",
            embedding_function=self.embeddings
        )

    # Required abstract methods from BaseStore
    def mget(self, keys: Sequence[str]) -> List[Optional[Document]]:
        """Batch-get documents by IDs."""
        return [self.collection.get(id=key) for key in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        """Batch-set documents."""
        for key, doc in key_value_pairs:
            self.collection.insert(doc.page_content, id=key)

    def mdelete(self, keys: Sequence[str]) -> None:
        """Batch-delete documents by IDs."""
        for key in keys:
            self.collection.delete(id=key)

    def yield_keys(self, prefix: str = None) -> Iterator[str]:
        """Iterate over keys matching a prefix."""
        for doc in self.collection.search(prefix):
            yield doc.id

    def get(self, field: str) -> List[Document]:
        """Get relevant documents using ChromaDB.

        Args:
            field (str): The field to search.

        Returns:
            List[Document]: The relevant documents.
        """
        return self.collection.search(field)

    def put(self, content: str) -> None:
        """Store new content in ChromaDB."""
        self.collection.add(ids=[str(uuid.uuid4())], documents=[content])

    async def batch(self, operations: Sequence[Tuple[str, tuple, str, Any]]) -> None:
        """Execute multiple operations in batch asynchronously.

        Args:
            operations (Sequence[Tuple[str, tuple, str, Any]]): The operations to execute.
        """
        await asyncio.gather(
            *[self._async_operation(*op) for op in operations]
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

    async def add_documents(self, docs: List[Document]) -> None:
        """Add multiple documents to the store.

        Args:
            docs (List[Document]): The documents to add.
        """
        for doc in docs:
            self.put(doc.page_content)

# Ensure max_size is parsed correctly
max_cache_size = parse_size(CACHING_SETTINGS.get('max_size', '100MB'))
