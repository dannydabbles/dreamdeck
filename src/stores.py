from langchain_core.documents import Document
from langgraph.store.base import BaseStore
from langchain_huggingface import HuggingFaceEmbeddings, Chroma
from typing import Dict, Any, List, Optional, Sequence, Tuple
import numpy as np
import os
import chainlit as cl
import asyncio

class VectorStore(BaseStore):
    """Custom vector store implementation using ChromaDB for persistent storage."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory="chroma_db"
        )
        
    def get(self, key: tuple, field: str) -> List[Document]:
        """Get relevant documents using ChromaDB."""
        try:
            thread_id = key[0] if key else cl.context.session.id
            if not thread_id:
                return []
                
            # Get relevant documents using embeddings
            try:
                query_embedding = self.embeddings.embed_query(field)
                docs = self.vectorstore.similarity_search(
                    query_embedding,
                    k=3
                )
                return [Document(page_content=d.page_content, metadata=d.metadata) for d in docs]
            except Exception as e:
                cl.logger.error(f"Embedding or search error: {e}")
                return []
                
        except Exception as e:
            cl.logger.error(f"Error in vector store get operation: {e}")
            return []
    
    def put(self, key: tuple, field: str, value: Dict[str, Any]) -> None:
        """Store new content in ChromaDB."""
        try:
            thread_id = key[0] if key else cl.context.session.id
            content = value.get("content", "")
            if not content:
                return
                
            # Create and add document
            doc = Document(
                page_content=content,
                metadata={
                    "thread_id": thread_id,
                    "timestamp": os.getenv("TIMESTAMP", ""),
                    "field": field
                }
            )
            self.vectorstore.add_documents([doc])
            self.vectorstore.persist()
            
        except Exception as e:
            cl.logger.error(f"Error storing document: {e}")
            raise

    async def abatch(self, operations: Sequence[Tuple[str, tuple, str, Any]]) -> None:
        """Execute multiple operations in batch asynchronously."""
        await asyncio.gather(*[
            asyncio.create_task(self._async_operation(op, key, field, value))
            for op, key, field, value in operations
        ])

    async def _async_operation(self, op: str, key: tuple, field: str, value: Any) -> None:
        """Helper method for async batch operations."""
        if op == "get":
            self.get(key, field)
        elif op == "put":
            self.put(key, field, value)
