import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage
from src.models import ChatState
from src.agents.orchestrator_agent import _decide_actions

@pytest.mark.asyncio
async def test_orchestrator_returns_actions():
    state = ChatState(messages=[HumanMessage(content="roll 2d6")], thread_id="test-thread-id")

    with (
        patch("src.agents.orchestrator_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke,
        patch("src.agents.orchestrator_agent.cl.user_session.get", new_callable=MagicMock) as mock_user_session_get,
    ):
        mock_ainvoke.return_value.content = '{"actions": ["roll", "write"]}'
        mock_user_session_get.return_value = {}
        actions = await _decide_actions(state)
        assert actions == ["roll", "write"]

@pytest.mark.asyncio
async def test_orchestrator_handles_invalid_json():
    state = ChatState(messages=[HumanMessage(content="tell me a story")], thread_id="test-thread-id")

    with (
        patch("src.agents.orchestrator_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke,
        patch("src.agents.orchestrator_agent.cl.user_session.get", new_callable=MagicMock) as mock_user_session_get,
    ):
        mock_ainvoke.return_value.content = 'not a json'
        mock_user_session_get.return_value = {}
        actions = await _decide_actions(state)
        assert actions == ["continue_story"]

@pytest.mark.asyncio
async def test_orchestrator_handles_exception():
    state = ChatState(messages=[HumanMessage(content="search dragons")], thread_id="test-thread-id")

    with (
        patch("src.agents.orchestrator_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock, side_effect=Exception("fail")),
        patch("src.agents.orchestrator_agent.cl.user_session.get", new_callable=MagicMock) as mock_user_session_get,
    ):
        mock_user_session_get.return_value = {}
        actions = await _decide_actions(state)
        assert actions == ["continue_story"]
