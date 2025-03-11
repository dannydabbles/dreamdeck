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
    
    # Mock the LLM's response to return "roll" explicitly
    with patch('src.agents.decision_agent.ChatOpenAI.agenerate', 
              new_callable=AsyncMock) as mock_agenerate:
        mock_generations = [Generation(text="The user wants to roll dice.")]
        # Correct structure: list of lists of Generation instances
        mock_result = LLMResult(generations=[[mock_generations[0]]])  
        mock_agenerate.return_value = mock_result
        
        result = await _decide_action(user_input, **mock_langgraph_context)
        assert result["name"] == "roll"

@pytest.mark.asyncio
async def test_web_search_integration(mock_langgraph_context):
    user_input = HumanMessage(content="search AI trends")
    with patch("src.agents.web_search_agent.requests.get", new_callable=AsyncMock) as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = {"organic_results": [{"snippet": "AI trends are evolving."}]}
        mock_get.return_value = mock_response
        
        # Force-enable web search for this test
        with patch.dict('src.config.__dict__', {'WEB_SEARCH_ENABLED': True}):  
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
