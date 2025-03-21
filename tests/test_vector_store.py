import pytest
from src.stores import VectorStore
from src.models import ChatState
from langchain_core.documents import Document

@pytest.mark.asyncio
async def test_vector_store_operations():
    store = VectorStore()
    test_doc = Document(page_content="Test document content")

    # Test storing documents
    await store.add_documents([test_doc])
    results = store.get("Test document")
    assert len(results) >= 1

    # Test retrieval
    query_results = store.get("document content")
    assert any("Test document" in doc.page_content for doc in query_results)
