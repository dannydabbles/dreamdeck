import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import chainlit as cl
from src.stores import VectorStore
from langchain_core.documents import Document
import asyncio
from chainlit.context import context_var, ChainlitContext

@pytest.fixture
def mock_chainlit_context():
    # Standardize context mocking using context_var
    mock_session = MagicMock()
    mock_session.thread_id = "test-thread-id"
    mock_context = MagicMock(spec=ChainlitContext)
    mock_context.session = mock_session
    mock_context.emitter = AsyncMock()

    token = context_var.set(mock_context)
    try:
        yield
    finally:
        context_var.reset(token)


@pytest.mark.asyncio
async def test_vector_store_operations(mock_chainlit_context):
    store = VectorStore()
    test_doc = Document(page_content="Test document content")

    await store.add_documents([test_doc])
    await asyncio.sleep(0.2)
    results = store.get("Test document")
    assert len(results) >= 1

    # Verify retrieval
    query_results = store.get("document content")
    assert any("Test document" in doc.page_content for doc in query_results)


@pytest.mark.asyncio
async def test_vector_store_put_and_get(mock_chainlit_context):
    store = VectorStore()
    await store.put("Hello world", "msg1", {"type": "human"})
    await asyncio.sleep(0.2)
    results = store.get("Hello")
    assert any("Hello" in doc.page_content for doc in results)
