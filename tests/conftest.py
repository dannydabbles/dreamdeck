import sys
import os

# Always set DREAMDECK_TEST_MODE=1 during tests to isolate ChromaDB
os.environ["DREAMDECK_TEST_MODE"] = "1"

# Patch chainlit.command to a dummy decorator during pytest collection and run
if (
    "pytest" in sys.modules
    or "PYTEST_CURRENT_TEST" in os.environ
    or "PYTEST_RUNNING" in os.environ
):
    try:
        import chainlit as cl

        def _noop_decorator(*args, **kwargs):
            def wrapper(func):
                return func
            return wrapper

        cl.command = _noop_decorator
        cl.profile = _noop_decorator  # Add this line to patch cl.profile during tests
        cl.step = _noop_decorator  # Add this line to patch cl.step during tests
    except ImportError:
        pass

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


import logging
import pytest

@pytest.fixture(scope="session", autouse=True)
def cleanup_logging():
    yield
    logging.shutdown()
