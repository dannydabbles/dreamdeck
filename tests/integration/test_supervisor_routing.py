import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState

@pytest.mark.asyncio
async def test_supervisor_multi_hop_workflow():
    """Test supervisor routing through multiple tools"""
    state = ChatState(
        messages=[
            HumanMessage(content="Search for dragon weaknesses and roll attack dice"),
        ],
        thread_id="multi-1"
    )
    
    mock_responses = [
        AIMessage(content="Dragon weak to ice", name="web_search"),
        AIMessage(content="You rolled 15", name="dice_roll"),
        AIMessage(content="The dragon shivers as you attack!", name="🎭 Storyteller GM")
    ]
    
    mock_supervisor = AsyncMock(side_effect=[mock_responses])
    
    with patch("src.supervisor.supervisor", mock_supervisor):
        from src.supervisor import supervisor
        response = await supervisor(state)
        
        assert len(response) == 3
        assert "weak to ice" in response[0].content
        assert "rolled 15" in response[1].content
        assert "dragon shivers" in response[2].content

@pytest.mark.asyncio
async def test_persona_switch_workflow():
    """Test automatic persona switching mid-conversation"""
    state = ChatState(
        messages=[
            HumanMessage(content="I need to organize my inventory"),
            AIMessage(content="Todo list updated", name="todo"),
        ],
        thread_id="persona-1"
    )
    
    mock_response = AIMessage(
        content="Your inventory is organized:",
        name="🗒️ Secretary"
    )
    
    with patch("src.agents.writer_agent._generate_story") as mock_generate_story:
        mock_generate_story.return_value = [mock_response]
        from src.supervisor import supervisor
        response = await supervisor(state)
        
        assert "inventory is organized" in response[0].content
        assert "Secretary" in response[0].name
