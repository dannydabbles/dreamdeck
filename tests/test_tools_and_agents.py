import pytest
from unittest.mock import AsyncMock, MagicMock
from langchain_core.outputs import Generation, LLMResult

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
        mock_result = LLMResult(generations=[[mock_generations]])
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
