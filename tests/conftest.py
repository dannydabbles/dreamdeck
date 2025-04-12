import sys
import os

# Always set DREAMDECK_TEST_MODE=1 during tests to isolate ChromaDB
os.environ["DREAMDECK_TEST_MODE"] = "1"

import pytest

# --- Centralized test monkeypatching for langgraph.func.task and agent registry ---

def _noop_decorator(*args, **kwargs):
    def wrapper(func):
        return func
    return wrapper

import asyncio

@pytest.fixture(autouse=True, scope="function")
def patch_task_and_registry(monkeypatch):
    # Patch langgraph.func.task to a context-preserving decorator for all tests
    def context_preserving_task(*args, **kwargs):
        def decorator(func):
            # Wrap the function to provide a dummy langgraph context if needed
            async def wrapper(*w_args, **w_kwargs):
                # Patch langgraph.config.get_config to not fail
                try:
                    import langgraph.config
                    monkeypatch.setattr(langgraph.config, "get_config", lambda: {})
                except ImportError:
                    pass
                return await func(*w_args, **w_kwargs)
            return wrapper
        return decorator
    monkeypatch.setattr("langgraph.func.task", context_preserving_task, raising=False)

    # Patch chainlit.command, .profile, .step to no-op
    try:
        import chainlit as cl
        cl.command = _noop_decorator
        cl.profile = _noop_decorator
        cl.step = _noop_decorator
    except ImportError:
        pass

    # Patch src.agents.registry.get_agent to always return the undecorated function if available
    import src.agents.registry as registry

    orig_get_agent = registry.get_agent

    def test_get_agent(name):
        agent = orig_get_agent(name)
        # If agent is a langgraph task, try to get the undecorated function
        if hasattr(agent, "_dice_roll"):
            return agent._dice_roll
        if hasattr(agent, "_generate_storyboard"):
            return agent._generate_storyboard
        if hasattr(agent, "_manage_todo"):
            return agent._manage_todo
        if hasattr(agent, "_knowledge"):
            return agent._knowledge
        if hasattr(agent, "_generate_story"):
            return agent._generate_story
        # If agent is a dummy or already a mock, just return it
        return agent

    monkeypatch.setattr(registry, "get_agent", test_get_agent)

    # Patch asyncio.run to work under pytest-asyncio (running event loop)
    orig_asyncio_run = asyncio.run

    def safe_asyncio_run(coro, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return orig_asyncio_run(coro, *args, **kwargs)
        # Already in a running loop (pytest-asyncio), so just await
        return loop.run_until_complete(coro) if not loop.is_running() else loop.create_task(coro)

    monkeypatch.setattr(asyncio, "run", safe_asyncio_run)

    yield

# Patch chainlit.command to a dummy decorator during pytest collection and run (legacy fallback)
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
        cl.profile = _noop_decorator
        cl.step = _noop_decorator
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
    # Shutdown all logging handlers to avoid "I/O operation on closed file" errors
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        handler.close()
        root_logger.removeHandler(handler)

    # Also explicitly close your named logger's handlers
    cl_logger = logging.getLogger("chainlit")
    for handler in cl_logger.handlers[:]:
        handler.close()
        cl_logger.removeHandler(handler)

    logging.shutdown()


import warnings


def pytest_sessionfinish(session, exitstatus):
    # Suppress asyncio "Task was destroyed but it is pending" warnings
    warnings.filterwarnings(
        "ignore", message=".*Task was destroyed but it is pending.*"
    )
    # Suppress unawaited coroutine warnings in test output
    warnings.filterwarnings(
        "ignore", message="coroutine '.*' was never awaited"
    )
