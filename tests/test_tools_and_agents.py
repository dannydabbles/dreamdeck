import os  # Import os at the top

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage
from langchain_core.outputs import Generation, LLMResult
from src.agents.decision_agent import _decide_action
from src.agents.web_search_agent import _web_search
from src.agents.writer_agent import _generate_story
from src.agents.storyboard_editor_agent import _generate_storyboard
from src.agents.dice_agent import _dice_roll

@pytest.fixture
def mock_langgraph_context():
    return {}

@pytest.mark.asyncio
async def test_decision_agent_roll_action(mock_langgraph_context):
    user_input = HumanMessage(content="roll 2d20")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    
    # Mock the LLM's response to return "roll" explicitly
    with patch('src.agents.decision_agent.ChatOpenAI.agenerate', 
              new_callable=AsyncMock) as mock_agenerate:
        mock_generations = [Generation(text="The user wants to roll dice.")]
        # Correct structure: list of lists of Generation instances
        mock_result = LLMResult(generations=[[mock_generations[0]]])  
        mock_agenerate.return_value = mock_result
        
        result = await _decide_action(state, **mock_langgraph_context)
        assert result[0].name == "dice_roll"

import src.config  # Import src.config at the top

@pytest.mark.asyncio
async def test_web_search_integration(mock_langgraph_context):
    user_input = HumanMessage(content="search AI trends")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    
    with patch("src.agents.web_search_agent.requests.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic_results": [{"snippet": "AI trends are evolving."}]}
        mock_get.return_value = mock_response
        
        result = await _web_search(state, **mock_langgraph_context)
        assert "AI trends are evolving" in result[0].content

@pytest.mark.asyncio
async def test_writer_agent_continuation(mock_langgraph_context):
    user_input = HumanMessage(content="Continue the adventure")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    result = await _generate_story(user_input.content, previous=state, **mock_langgraph_context)
    assert result.strip()

@pytest.mark.asyncio
async def test_storyboard_editor_agent(mock_langgraph_context):
    user_input = HumanMessage(content="Generate a storyboard")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    with patch("src.image_generation.generate_image_async", new_callable=AsyncMock) as mock_generate_image:
        mock_generate_image.return_value = b"image_bytes"
        result = await _generate_storyboard(state, **mock_langgraph_context)
        assert result[0].content.strip()

@pytest.mark.asyncio
async def test_dice_agent():
    user_input = HumanMessage(content="roll d20")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    result = await _dice_roll(state)
    assert "rolled" in result[0].content.lower()
