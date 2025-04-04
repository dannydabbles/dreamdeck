import pytest
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.fixture
def mock_chainlit_context():
    mock_session = MagicMock()
    mock_session.thread_id = "test-thread-id"
    mock_context = MagicMock()
    mock_context.session = mock_session
    mock_context.emitter = AsyncMock()  # Use AsyncMock for async compatibility

    with patch("chainlit.context", mock_context):
        yield

import pytest
from src.config import (
    config,
    DefaultsConfig,
    LlmConfig,
    DiceConfig,
    FeatureConfig,
    DecisionAgentConfig,
    WriterAgentConfig,
    StoryboardEditorAgentConfig,
    AgentsConfig,
)


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("APP_FEATURES_WEB_SEARCH", "true")  # Explicitly set
    monkeypatch.setenv("APP_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("APP_SERPAPI_KEY", "test-serp-api-key")
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
