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
    original_context = cl.context
    mock_session = MagicMock(spec=cl.context.session)
    mock_session.thread_id = "test-thread-id"
    mock_context = MagicMock(spec=cl.context)
    mock_context.session = mock_session
    cl.context = mock_context
    yield

def test_vector_store_operations(mock_chainlit_context):  # <-- USE THE FIXTURE
    store = VectorStore()
    test_doc = Document(page_content="Test document content")

    # Test storing documents
    asyncio.run(store.add_documents([test_doc]))  # <-- ADJUST FOR ASYNC
    results = store.get("Test document")
    assert len(results) >= 1

    # Test retrieval
    query_results = store.get("document content")
    assert any("Test document" in doc.page_content for doc in query_results)
