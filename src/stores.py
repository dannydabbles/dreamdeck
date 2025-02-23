from langchain_core.documents import Document
from langgraph.store.base import BaseStore
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict, Any, List, Optional, Sequence, Tuple
import numpy as np
from chainlit.types import ThreadDict
import chainlit as cl
import asyncio

class VectorStore(BaseStore):
    """Custom vector store implementation using Chainlit's thread history."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        
    def _get_thread_history(self, thread_id: str) -> Optional[ThreadDict]:
        """Get thread history from Chainlit."""
        return cl.user_session.get("thread")
        
    def _convert_to_documents(self, thread: ThreadDict) -> List[Document]:
        """Convert thread history to documents."""
        docs = []
        for step in thread.get("steps", []):
            if step.get("type") in ["user_message", "ai_message"]:
                docs.append(Document(
                    page_content=step["output"],
                    metadata={
                        "type": step["type"],
                        "created_at": step["createdAt"],
                        "step_id": step["id"]
                    }
                ))
        return docs

    def get(self, key: tuple, field: str) -> List[Document]:
        """Get relevant documents from thread history."""
        try:
            thread_id = key[0] if key else cl.context.session.id
            thread = self._get_thread_history(thread_id)
            if not thread:
                return []
                
            docs = self._convert_to_documents(thread)
            if not docs:
                return []
                
            # Get relevant documents using embeddings
            query_embedding = self.embeddings.embed_query(field)
            doc_embeddings = self.embeddings.embed_documents([d.page_content for d in docs])
            
            # Cosine similarity search with metadata preservation
            similarities = [np.dot(query_embedding, doc_emb) for doc_emb in doc_embeddings]
            most_similar = sorted(zip(similarities, docs), reverse=True)[:3]
            
            return [doc for _, doc in most_similar]
        except Exception as e:
            cl.logger.error(f"Error in vector store get operation: {e}")
            return []
    
    def put(self, key: tuple, field: str, value: Dict[str, Any]) -> None:
        """Store new content in thread history."""
        thread_id = key[0] if key else cl.context.session.id
        
        # Store as element in thread
        if "content" in value:
            element = cl.Element(
                type="text",
                content=value["content"],
                metadata=value.get("metadata", {})
            )
            cl.user_session.set(f"element_{thread_id}", element)

    def batch(self, operations: Sequence[Tuple[str, tuple, str, Any]]) -> None:
        """Execute multiple operations in batch."""
        for op, key, field, value in operations:
            if op == "get":
                self.get(key, field)
            elif op == "put":
                self.put(key, field, value)

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
