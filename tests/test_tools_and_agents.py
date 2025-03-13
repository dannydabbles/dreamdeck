import os  # Import os at the top

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import Generation
from src.models import ChatState  # Import ChatState
from src.agents.decision_agent import _decide_action
from src.agents.web_search_agent import _web_search
from src.agents.writer_agent import _generate_story
from src.agents.storyboard_editor_agent import _generate_storyboard
from src.agents.dice_agent import _dice_roll

@pytest.mark.asyncio
async def test_decision_agent_roll_action():
    user_input = HumanMessage(content="roll 2d20")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    
    # Mock the LLM's response to return "roll" explicitly
    with patch('src.agents.decision_agent.ChatOpenAI.invoke', 
              new_callable=AsyncMock) as mock_agenerate:
        mock_result = AIMessage(content="roll")  
        mock_agenerate.return_value = mock_result
        
        result = await _decide_action(state)
        assert result[0].name == "dice_roll"

import src.config  # Import src.config at the top

@pytest.mark.asyncio
async def test_web_search_integration():
    user_input = HumanMessage(content="search AI trends")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    
    with patch("src.agents.web_search_agent.requests.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic_results": [{"snippet": "AI trends are evolving."}]}
        mock_get.return_value = mock_response
        
        result = await _web_search(state)
        assert "AI trends are evolving" in result[0].content

@pytest.mark.asyncio
async def test_dice_agent():
    user_input = HumanMessage(content="roll d20")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")
    result = await _dice_roll(state)
    assert "roll" in result[0].content.lower()
