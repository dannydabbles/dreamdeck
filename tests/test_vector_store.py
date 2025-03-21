import pytest
from unittest.mock import MagicMock
import chainlit as cl
from src.stores import VectorStore  # Import VectorStore
from langchain_core.documents import Document  # Import Document
import asyncio  # Import asyncio

@pytest.fixture
def mock_chainlit_context():
    original_context = cl.Context.get_current()
    mock_session = MagicMock(spec=cl.context.session)
    mock_session.thread_id = "test-thread-id"
    mock_context = MagicMock(spec=cl.context)
    mock_context.session = mock_session
    cl.context = mock_context
    yield

@pytest.fixture
def mock_chainlit_context():
    mock_session = MagicMock()
    mock_session.thread_id = "test-thread-id"  # Set a valid thread ID
    mock_context = MagicMock()
    mock_context.session = mock_session

    # Use patch to replace the global context
    with patch("chainlit.context", mock_context):
        yield

def test_vector_store_operations(mock_chainlit_context):  # Use the fixed fixture
    store = VectorStore()
    test_doc = Document(page_content="Test document content")

    # Explicitly run async methods with asyncio.run (or ensure event loop)
    asyncio.run(store.add_documents([test_doc]))
    results = store.get("Test document")
    assert len(results) >= 1

    # Verify retrieval
    query_results = store.get("document content")
    assert any("Test document" in doc.page_content for doc in query_results)
