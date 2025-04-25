import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import chainlit as cl
import pytest
from chainlit.context import ChainlitContext, context_var
from langchain_core.documents import Document

from src.stores import VectorStore


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

    # Add a unique test document to avoid interference from other docs
    unique_text = "UniqueHelloTestString"
    await store.put(unique_text, "unique-msg-id", {"type": "human"})
    await asyncio.sleep(0.2)

    results = store.get("UniqueHelloTestString")

    print("VectorStore get('UniqueHelloTestString') returned:")
    for doc in results:
        print(f"- {doc.page_content!r}")

    # If no match, print all docs in the collection for debugging
    if not any("UniqueHelloTestString" in doc.page_content for doc in results):
        all_results = store.get("")  # empty query returns top docs
        print("Full collection contents:")
        for doc in all_results:
            print(f"- {doc.page_content!r}")

    # Assert the unique string is found
    assert any("uniquehelloteststring" in doc.page_content.lower() for doc in results)
