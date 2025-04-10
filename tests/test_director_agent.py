import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage
from src.models import ChatState
from src.agents.director_agent import _direct_actions


@pytest.mark.asyncio
async def test_director_returns_actions():
    state = ChatState(
        messages=[HumanMessage(content="roll 2d6")], thread_id="test-thread-id"
    )

    with (
        patch(
            "src.agents.director_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke,
        patch(
            "src.agents.director_agent.cl.user_session.get", new_callable=MagicMock
        ) as mock_user_session_get,
    ):
        mock_ainvoke.return_value.content = '{"actions": ["roll", {"action": "knowledge", "type": "character"}, "write"]}'
        mock_user_session_get.return_value = {}
        actions = await _direct_actions(state)
        assert actions == [
            "roll",
            {"action": "knowledge", "type": "character"},
            "write",
        ]


@pytest.mark.asyncio
async def test_director_handles_invalid_json():
    state = ChatState(
        messages=[HumanMessage(content="tell me a story")], thread_id="test-thread-id"
    )

    with (
        patch(
            "src.agents.director_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke,
        patch(
            "src.agents.director_agent.cl.user_session.get", new_callable=MagicMock
        ) as mock_user_session_get,
    ):
        mock_ainvoke.return_value.content = "not a json"
        mock_user_session_get.return_value = {}
        actions = await _direct_actions(state)
        assert actions == ["continue_story"]


@pytest.mark.asyncio
async def test_director_handles_exception():
    state = ChatState(
        messages=[HumanMessage(content="search dragons")], thread_id="test-thread-id"
    )

    with (
        patch(
            "src.agents.director_agent.ChatOpenAI.ainvoke",
            new_callable=AsyncMock,
            side_effect=Exception("fail"),
        ),
        patch(
            "src.agents.director_agent.cl.user_session.get", new_callable=MagicMock
        ) as mock_user_session_get,
    ):
        mock_user_session_get.return_value = {}
        actions = await _direct_actions(state)
        assert actions == ["continue_story"]


@pytest.mark.asyncio
async def test_director_handles_invalid_action_format():
    state = ChatState(
        messages=[HumanMessage(content="tell me a story")], thread_id="test-thread-id"
    )

    with (
        patch(
            "src.agents.director_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke,
        patch(
            "src.agents.director_agent.cl.user_session.get", new_callable=MagicMock
        ) as mock_user_session_get,
    ):
        mock_ainvoke.return_value.content = '{"actions": ["roll", 123]}'
        mock_user_session_get.return_value = {}
        actions = await _direct_actions(state)
        assert actions == ["continue_story"]
