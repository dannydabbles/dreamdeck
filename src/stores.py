from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.store.base import BaseStore

class VectorStore(BaseStore):
    """Custom vector store implementation."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        self.vectorstore = Chroma(
            collection_name="dreamdeck",
            embedding_function=self.embeddings
        )
    
    def get(self, key: tuple, field: str):
        """Get a value from the store."""
        return self.vectorstore.similarity_search(field, k=3)
    
    def put(self, key: tuple, field: str, value: dict):
        """Put a value in the store."""
        self.vectorstore.add_texts([value["content"]], metadatas=[value["metadata"]])
