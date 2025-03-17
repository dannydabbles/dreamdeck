import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.event_handlers import load_knowledge_documents
from src.stores import VectorStore
from langchain_core.documents import Document
from src.config import KNOWLEDGE_DIRECTORY

@pytest.mark.asyncio
async def test_load_knowledge_documents(tmp_path, monkeypatch):
    # Mock the knowledge directory path
    mock_dir = tmp_path / "knowledge"
    mock_dir.mkdir()
    monkeypatch.setattr("src.event_handlers.KNOWLEDGE_DIRECTORY", str(mock_dir))
    
    # Create test files
    (mock_dir / "test.txt").write_text("Sample text content")
    (mock_dir / "subfolder").mkdir()
    (mock_dir / "subfolder" / "test.pdf").touch()  # Empty PDF for coverage
    
    # Mock dependencies
    vector_store_mock = MagicMock(spec=VectorStore)
    vector_store_mock.add_documents = AsyncMock()
    monkeypatch.setattr("src.event_handlers.cl.user_session.get", lambda _: vector_store_mock)
    
    with patch("src.event_handlers._load_document") as load_doc_mock,\
         patch("src.event_handlers.RecursiveCharacterTextSplitter") as splitter_mock:
        
        # Return dummy documents and splits
        load_doc_mock.side_effect = lambda _: [Document(page_content="Loaded content")]
        splitter_mock.return_value.split_documents.return_value = [
            Document(page_content="Chunk 1"),
            Document(page_content="Chunk 2")
        ]
        
        # Run the knowledge loader
        await load_knowledge_documents()
        
        # Verify operations
        assert load_doc_mock.call_count == 2  # Both txt and pdf
        assert splitter_mock().split_documents.called
        
        # Verify vector store receives chunks
        add_call = vector_store_mock.add_documents.call_args_list[0]
        added_docs = add_call.args[0]
        assert len(added_docs) == 2  # Two chunks from each file
        assert "Chunk 1" in [doc.page_content for doc in added_docs]
