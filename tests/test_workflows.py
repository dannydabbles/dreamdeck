import pytest
from unittest.mock import AsyncMock, patch
from src.workflows import chat_workflow
from src.state import ChatState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.stores import VectorStore

@pytest.fixture
def mock_chat_state():
    return ChatState(
        messages=[HumanMessage(content="Hello, GM!")]
    )

@pytest.mark.asyncio
async def test_chat_workflow(mock_chat_state):
    with patch("src.agents.decision_agent.decide_action", new_callable=AsyncMock) as mock_decide_action, \
         patch("src.agents.dice_agent.dice_roll", new_callable=AsyncMock) as mock_dice_roll, \
         patch("src.agents.web_search_agent.web_search", new_callable=AsyncMock) as mock_web_search, \
         patch("src.agents.writer_agent.generate_story", new_callable=AsyncMock) as mock_generate_story, \
         patch("src.agents.storyboard_editor_agent.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_editor_agent:

        mock_decide_action.return_value = {"name": "continue_story"}
        mock_generate_story.return_value = "The adventure continues..."
        mock_storyboard_editor_agent.return_value = "Storyboard generated."

        store = VectorStore()
        updated_state = await chat_workflow([HumanMessage(content="Continue the adventure")], store, previous=mock_chat_state)

        assert len(updated_state.messages) > len(mock_chat_state.messages)
        assert isinstance(updated_state.messages[-1], AIMessage)
        assert "The adventure continues..." in updated_state.messages[-1].content
