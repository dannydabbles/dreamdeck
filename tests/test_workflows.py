import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.models import ChatState
from src.workflows import _chat_workflow

@pytest.fixture
def mock_chat_state():
    return ChatState(
        messages=[AIMessage(content="Hello, I'm the GM!", name="Game Master")],
        thread_id="test-thread-id",
        error_count=0,
    )

@pytest.mark.asyncio
async def test_chat_workflow(mock_chat_state):
    from unittest.mock import MagicMock, AsyncMock

    # Mock the @task decorator globally to disable context requirements
    with patch("src.agents.decision_agent.task", new=lambda f: f), \
         patch("src.agents.dice_agent.task", new=lambda f: f), \
         patch("src.agents.web_search_agent.task", new=lambda f: f), \
         patch("src.agents.writer_agent.task", new=lambda f: f), \
         patch("src.agents.storyboard_editor_agent.task", new=lambda f: f):

        with (
            patch("src.agents.decision_agent.decide_action", new_callable=AsyncMock) as mock_decide_action,
            patch("src.agents.dice_agent.dice_roll", new_callable=AsyncMock) as mock_dice_roll,
            patch("src.agents.web_search_agent.web_search", new_callable=AsyncMock) as mock_web_search,
            patch("src.agents.writer_agent.generate_story", new_callable=AsyncMock) as mock_write_story,
            patch("src.agents.storyboard_editor_agent.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_agent,
        ):
            # Mock agent outputs
            mock_decide_action.return_value = [AIMessage(name="continue_story", content="The adventure continues...", additional_kwargs={})]
            mock_write_story.return_value = [AIMessage(content="The adventure continues...", name="game_master", additional_kwargs={})]
            mock_storyboard_agent.return_value = []

            initial_state = mock_chat_state
            new_messages = [HumanMessage(content="Continue the adventure")]

            updated_state = await _chat_workflow(new_messages, previous=initial_state)

            assert len(updated_state.messages) == len(initial_state.messages) + 1
            assert any(
                msg.content == "The adventure continues..." 
                for msg in updated_state.messages 
                if isinstance(msg, AIMessage)
            )
