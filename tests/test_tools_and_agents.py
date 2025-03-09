import pytest
from src.agents.decision_agent import decision_agent
from src.agents.writer_agent import writer_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.dice_agent import dice_roll  # Import the tool, not the agent
from src.agents.web_search_agent import web_search  # Import the tool, not the agent
from src.agents.web_search_agent import web_search_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import json
import asyncio
from unittest.mock import patch, MagicMock
from uuid import uuid4  # Import uuid4

# Mock the config object with the required features
@pytest.fixture
def mock_config():
    class MockConfig:
        features = {
            'dice_rolling': True,
            'web_search': False,
            'image_generation': True
        }
    return MockConfig()

@pytest.mark.asyncio
async def test_decision_agent_roll_action(mock_config):
    with patch('src.agents.decision_agent.config', mock_config), \
         patch('src.agents.decision_agent.decide_action', return_value={"name": "roll", "args": {}}), \
         patch('src.agents.dice_agent.dice_roll', return_value=ToolMessage(content="🎲 You rolled 15 on a 20-sided die.", tool_call_id=str(uuid4()), name="dice_roll")):
        response = await decision_agent.ainvoke(
            [HumanMessage(content="roll 2d20")],
            config={"configurable": {"thread_id": "test-thread-123"}}
        )
    assert isinstance(response, AIMessage)
    assert "🎲 You rolled" in response.content

@pytest.mark.asyncio
async def test_decision_agent_search_action(mock_config):
    with patch('src.agents.decision_agent.config', mock_config), \
         patch('src.agents.decision_agent.decide_action', return_value={"name": "search", "args": {}}), \
         patch('src.agents.web_search_agent.web_search', return_value=ToolMessage(content="Web search result: AI is a field of computer science.", tool_call_id=str(uuid4()), name="web_search")):
        response = await decision_agent.ainvoke(
            [HumanMessage(content="search for information on AI")],
            config={"configurable": {"thread_id": "test-thread-456"}}
        )
    assert isinstance(response, AIMessage)
    assert "Web search" in response.content

@pytest.mark.asyncio
async def test_decision_agent_story_action(mock_config):
    with patch('src.agents.decision_agent.config', mock_config), \
         patch('src.agents.decision_agent.decide_action', return_value={"name": "continue_story", "args": {}}):
        response = await decision_agent.ainvoke(
            [HumanMessage(content="continue the story")],
            config={"configurable": {"thread_id": "test-thread-789"}}
        )
    assert isinstance(response, AIMessage)
    assert "continue_story" in response.content

@pytest.mark.asyncio
async def test_dice_roll_agent(mock_config):
    with patch('src.agents.dice_agent.config', mock_config), \
         patch('src.agents.dice_agent.dice_roll', return_value=ToolMessage(content="🎲 You rolled 15 on a 20-sided die.", tool_call_id=str(uuid4()), name="dice_roll")):
        response = await dice_roll_agent.ainvoke(
            [HumanMessage(content="roll 2d20")],
            config={"configurable": {"thread_id": "test-thread-101"}}
        )
    assert isinstance(response, AIMessage)
    assert "🎲 You rolled" in response.content

@pytest.mark.asyncio
async def test_web_search_agent(mock_config):
    with patch('src.agents.web_search_agent.config', mock_config), \
         patch('src.agents.web_search_agent.web_search', return_value=ToolMessage(content="Web search result: AI is a field of computer science.", tool_call_id=str(uuid4()), name="web_search")):
        response = await web_search_agent.ainvoke(
            [HumanMessage(content="search for AI information")],
            config={"configurable": {"thread_id": "test-thread-102"}}
        )
    assert isinstance(response, AIMessage)
    assert "Web search" in response.content

@pytest.mark.asyncio
async def test_writer_agent(mock_config):
    with patch('src.agents.writer_agent.config', mock_config):
        response = await writer_agent.ainvoke(
            [HumanMessage(content="continue the story")],
            config={"configurable": {"thread_id": "test-thread-103"}}
        )
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0

@pytest.mark.asyncio
async def test_storyboard_editor_agent(mock_config):
    with patch('src.agents.storyboard_editor_agent.config', mock_config):
        response = await storyboard_editor_agent.ainvoke(
            [HumanMessage(content="generate a storyboard")],
            config={"configurable": {"thread_id": "test-thread-104"}}
        )
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0
