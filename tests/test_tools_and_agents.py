import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from src.state import ChatState
from src.state_graph import chat_workflow
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from src.agents.decision_agent import _decide_action
from src.agents.dice_agent import _dice_roll
from src.agents.web_search_agent import _web_search
from src.agents.writer_agent import _generate_story
from src.agents.storyboard_editor_agent import _generate_storyboard

@pytest.fixture
def mock_langgraph_context():
    return {
        "store": MagicMock(),
        "previous": ChatState(),
        "config": {"configurable": {"thread_id": "test-thread-id"}}
    }

@pytest.mark.asyncio
async def test_decision_agent_roll_action(mock_langgraph_context):
    user_input = HumanMessage(content="roll 2d20")
    result = await _decide_action(user_input, **mock_langgraph_context)
    assert result["name"] == "roll"

@pytest.mark.asyncio
async def test_web_search_integration(mock_langgraph_context):
    user_input = HumanMessage(content="search AI trends")
    result = await _web_search(user_input, **mock_langgraph_context)
    assert "error" not in result["name"]

@pytest.mark.asyncio
async def test_writer_agent_continuation(mock_langgraph_context):
    user_input = HumanMessage(content="Continue the adventure")
    result = await _generate_story(user_input, **mock_langgraph_context)
    assert result.content.strip()

@pytest.mark.asyncio
async def test_storyboard_editor_agent(mock_langgraph_context):
    user_input = HumanMessage(content="Generate a storyboard")
    result = await _generate_storyboard(user_input, **mock_langgraph_context)
    assert result.content.strip()

@pytest.mark.asyncio
async def test_dice_agent():
    result = await _dice_roll("d20")
    assert "rolled" in result.content.lower()
