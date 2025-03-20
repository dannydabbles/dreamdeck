import pytest
from unittest.mock import AsyncMock, patch
from src.workflows import _chat_workflow
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.stores import VectorStore
from unittest.mock import MagicMock


@pytest.fixture
def mock_chat_state():
    return ChatState(
        messages=[AIMessage(content="Hello, I'm the GM!", name="Game Master")],
        thread_id="test-thread-id",
        error_count=0,
    )


@pytest.mark.asyncio
async def test_chat_workflow(mock_chat_state):
    with (
        patch("langgraph.config.get_config", return_value={}),  # Fix for missing context
        patch("src.agents.decision_agent.decide_action", new_callable=AsyncMock) as mock_decide_action,
        patch("src.agents.dice_agent.dice_roll", new_callable=AsyncMock) as mock_dice_roll,
        patch("src.agents.web_search_agent.web_search", new_callable=AsyncMock) as mock_web_search,
        patch("src.agents.writer_agent.generate_story", new_callable=AsyncMock) as mock_generate_story,
        patch("src.agents.storyboard_editor_agent.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_editor_agent,
    ):
        # Mock decision_agent's response
        mock_decide_action.return_value = [AIMessage(
            name="continue_story",
            content="The adventure continues...",
            additional_kwargs={}
        )]
        mock_generate_story.return_value = [AIMessage(
            content="The adventure continues...",
            name="game_master",
            additional_kwargs={}
        )]
        mock_storyboard_editor_agent.return_value = []

        # Add a human message to the state
        initial_state = mock_chat_state
        new_messages = [HumanMessage(content="Continue the adventure")]

        # Run the chat workflow
        updated_state = await _chat_workflow(new_messages, previous=initial_state)

        # Assertions
        assert len(updated_state.messages) > len(mock_chat_state.messages)
        assert any(
            isinstance(msg, AIMessage) and "The adventure continues..." in msg.content
            for msg in updated_state.messages
        )

    with (
        patch("langgraph.config.get_config", return_value={}),
        patch(
            "src.agents.decision_agent.decide_action", new_callable=AsyncMock
        ) as mock_decide_action,
        patch(
            "src.agents.dice_agent.dice_roll", new_callable=AsyncMock
        ) as mock_dice_roll,
        patch(
            "src.agents.web_search_agent.web_search", new_callable=AsyncMock
        ) as mock_web_search,
        patch(
            "src.agents.writer_agent.generate_story", new_callable=AsyncMock
        ) as mock_generate_story,
        patch(
            "src.agents.storyboard_editor_agent.storyboard_editor_agent",
            new_callable=AsyncMock,
        ) as mock_storyboard_editor_agent,
    ):

        mock_decide_action.return_value = [
            AIMessage(name="continue_story", content="The adventure continues...")
        ]
        mock_generate_story.return_value = "The adventure continues..."
        mock_storyboard_editor_agent.return_value = []

        store = MagicMock()  # Use MagicMock instead of real VectorStore
        state = mock_chat_state
        state.messages.append(HumanMessage(content="Continue the adventure"))
        updated_state = await _chat_workflow(state.messages, store, previous=state)

        assert len(updated_state.messages) > len(mock_chat_state.messages)
        assert any(
            isinstance(msg, AIMessage) and "The adventure continues..." in msg.content
            for msg in updated_state.messages
        )
        assert "The adventure continues..." in updated_state.messages[-1].content
