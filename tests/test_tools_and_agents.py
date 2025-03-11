import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from src.state import ChatState
from src.workflows import chat_workflow
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from src.agents.decision_agent import _decide_action
from src.agents.dice_agent import _dice_roll
from src.agents.web_search_agent import _web_search
from src.agents.writer_agent import _generate_story
from src.agents.storyboard_editor_agent import _generate_storyboard

@pytest.fixture
def mock_langgraph_context():
    return {}

@pytest.mark.asyncio
async def test_decision_agent_roll_action(mock_langgraph_context):
    user_input = HumanMessage(content="roll 2d20")
    result = await _decide_action(user_input, **mock_langgraph_context)
    assert result["name"] == "roll"

@pytest.mark.asyncio
async def test_web_search_integration(mock_langgraph_context):
    user_input = HumanMessage(content="search AI trends")
    with patch("src.agents.web_search_agent.requests.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic_results": [{"snippet": "AI trends are evolving."}]}
        mock_get.return_value = mock_response
        result = await _web_search(user_input.content, **mock_langgraph_context)
        assert "AI trends are evolving" in result.content

@pytest.mark.asyncio
async def test_writer_agent_continuation(mock_langgraph_context):
    user_input = HumanMessage(content="Continue the adventure")
    result = await _generate_story(user_input.content, **mock_langgraph_context)
    assert result.strip()

@pytest.mark.asyncio
async def test_storyboard_editor_agent(mock_langgraph_context):
    user_input = HumanMessage(content="Generate a storyboard")
    with patch("src.image_generation.generate_image_async", new_callable=AsyncMock) as mock_generate_image:
        mock_generate_image.return_value = b"image_bytes"
        result = await _generate_storyboard(user_input.content, **mock_langgraph_context)
        assert result.strip()

@pytest.mark.asyncio
async def test_dice_agent():
    result = await _dice_roll("d20")
    assert "rolled" in result.content.lower()
