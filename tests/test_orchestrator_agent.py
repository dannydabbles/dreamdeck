import pytest
from unittest.mock import patch, AsyncMock
from langchain_core.messages import HumanMessage
from src.models import ChatState
from src.agents.orchestrator_agent import _decide_actions

@pytest.mark.asyncio
async def test_orchestrator_returns_actions():
    state = ChatState(messages=[HumanMessage(content="roll 2d6")], thread_id="test-thread-id")

    with patch("src.agents.orchestrator_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value.content = '{"actions": ["roll", "write"]}'
        actions = await _decide_actions(state)
        assert actions == ["roll", "write"]

@pytest.mark.asyncio
async def test_orchestrator_handles_invalid_json():
    state = ChatState(messages=[HumanMessage(content="tell me a story")], thread_id="test-thread-id")

    with patch("src.agents.orchestrator_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke:
        mock_ainvoke.return_value.content = 'not a json'
        actions = await _decide_actions(state)
        assert actions == ["continue_story"]

@pytest.mark.asyncio
async def test_orchestrator_handles_exception():
    state = ChatState(messages=[HumanMessage(content="search dragons")], thread_id="test-thread-id")

    with patch("src.agents.orchestrator_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock, side_effect=Exception("fail")):
        actions = await _decide_actions(state)
        assert actions == ["continue_story"]
