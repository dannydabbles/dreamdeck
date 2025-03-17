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
    with patch(
        "src.agents.decision_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
    ) as mock_ainvoke:
        mock_result = AIMessage(content="roll", name="dice_roll")
        mock_ainvoke.return_value = mock_result

        result = await _decide_action(state)
        assert result[0].name == "dice_roll"


import src.config  # Import src.config at the top


@pytest.mark.asyncio
async def test_web_search_integration():
    user_input = HumanMessage(content="search AI trends")
    state = ChatState(messages=[user_input], thread_id="test-thread-id")

    with (
        patch(
            "src.agents.web_search_agent.requests.get", new_callable=MagicMock
        ) as mock_get,
        patch(
            "src.agents.web_search_agent.cl.Message", new_callable=MagicMock
        ) as mock_cl_message,
        patch(
            "src.agents.decision_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke,
    ):
        # Mock the HTTP GET response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic_results": [{"snippet": "AI trends are evolving."}]
        }
        mock_get.return_value = mock_response

        # Mock cl.Message and its send() method
        mock_cl_instance = AsyncMock()  # Use AsyncMock
        mock_cl_message.return_value = mock_cl_instance
        mock_cl_instance.send.return_value = None  # Simulate successful send

        # Mock the LLM's response to return "AI trends" explicitly
        mock_result = AIMessage(content="AI trends", name="web_search")
        mock_ainvoke.return_value = mock_result

        # Run the function under test
        result = await _web_search(state)

        # Verify the mocked send was called with correct args
        mock_cl_message.assert_called_once_with(
            content=f'**Search Results for "AI trends":**\n\n1. AI trends are evolving.',
            parent_id=None,
        )
        mock_cl_instance.send.assert_called_once()

        # Assert the result content includes the expected snippet
        assert "AI trends are evolving" in result[0].content


@pytest.mark.asyncio
async def test_dice_agent():
    with patch(
        "src.agents.decision_agent.ChatOpenAI.ainvoke", new_callable=MagicMock
    ) as mock_ainvoke:
        mock_result = AIMessage(content="roll", name="dice_roll")
        mock_ainvoke.return_value = mock_result
        user_input = HumanMessage(content="roll d20")
        state = ChatState(messages=[user_input], thread_id="test-thread-id")
        result = await _dice_roll(state)
        assert "roll" in result[0].content.lower()
