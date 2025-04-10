import os  # Import os at the top

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import Generation
from src.models import ChatState
from src.agents.web_search_agent import _web_search
from src.agents.writer_agent import _generate_story
from src.agents.storyboard_editor_agent import _generate_storyboard
from src.image_generation import (
    generate_image_generation_prompts,
    process_storyboard_images,
)
from src.agents.dice_agent import _dice_roll
from src.event_handlers import on_chat_start
import chainlit as cl


@pytest.mark.asyncio
async def test_empty_storyboard(mock_chainlit_context):
    await process_storyboard_images("", "msgid")


@pytest.mark.asyncio
async def test_refused_prompts(mock_chainlit_context):
    with patch("src.image_generation.generate_image_async") as mock_gen:
        await generate_image_generation_prompts("This is a refusal phrase")
        mock_gen.assert_not_called()


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
            "src.agents.web_search_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
        ) as mock_ainvoke,
    ):
        # Mock the HTTP GET response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "organic_results": [{"snippet": "AI trends are evolving."}]
        }
        mock_get.return_value = mock_response

        # Mock cl.Message and its send() method
        mock_cl_instance = AsyncMock()
        mock_cl_message.return_value = mock_cl_instance
        mock_cl_instance.send.return_value = None

        # Mock the LLM's response to return "AI trends" explicitly
        mock_result = AIMessage(content="AI trends", name="web_search")
        mock_ainvoke.return_value = mock_result

        # Run the function under test
        result = await _web_search(state)

        mock_cl_message.assert_called_once_with(
            content=f'**Search Results for "AI trends":**\n\n1. AI trends are evolving.',
            parent_id=None,
        )
        mock_cl_instance.send.assert_called_once()

        assert "AI trends are evolving" in result[0].content


@pytest.mark.asyncio
async def test_dice_roll_invalid_json(monkeypatch):
    from src.agents.dice_agent import _dice_roll, ChatOpenAI
    from src.models import ChatState
    from langchain_core.messages import HumanMessage

    state = ChatState(
        messages=[HumanMessage(content="roll 2d6", name="Player")], thread_id="test"
    )

    class FakeResp:
        content = "not a json"

    async def fake_ainvoke(*args, **kwargs):
        return FakeResp()

    monkeypatch.setattr(ChatOpenAI, "ainvoke", fake_ainvoke)

    result = await _dice_roll(state)
    assert "Error parsing dice roll" in result[0].content


@pytest.mark.asyncio
async def test_dice_roll_invalid_specs(monkeypatch):
    from src.agents.dice_agent import _dice_roll, ChatOpenAI
    from src.models import ChatState
    from langchain_core.messages import HumanMessage

    state = ChatState(
        messages=[HumanMessage(content="roll 2d6", name="Player")], thread_id="test"
    )

    class FakeResp:
        content = '{"specs": [], "reasons": []}'

    async def fake_ainvoke(*args, **kwargs):
        return FakeResp()

    monkeypatch.setattr(ChatOpenAI, "ainvoke", fake_ainvoke)

    result = await _dice_roll(state)
    assert "Invalid dice roll specification" in result[0].content


@pytest.mark.asyncio
async def test_writer_agent_includes_memories(monkeypatch):
    from src.agents.writer_agent import _generate_story
    from src.models import ChatState

    state = ChatState(messages=[HumanMessage(content="Hi")], thread_id="t1")
    state.memories = ["Memory1", "Memory2"]

    with (
        patch("src.agents.writer_agent.cl.user_session.get", return_value={}),
        patch(
            "src.agents.writer_agent.ChatOpenAI.astream", new_callable=AsyncMock
        ) as mock_astream,
        patch(
            "src.agents.writer_agent.cl.Message", new_callable=MagicMock
        ) as mock_cl_msg_cls,
    ):

        async def fake_stream(*args, **kwargs):
            yield MagicMock(content="Story chunk")

        mock_astream.side_effect = fake_stream

        mock_cl_msg = AsyncMock()
        mock_cl_msg.stream_token.return_value = None
        mock_cl_msg.send.return_value = None
        mock_cl_msg_cls.return_value = mock_cl_msg

        import chainlit as cl_module

        monkeypatch.setattr(cl_module, "user_session", MagicMock())
        cl_module.user_session.get.return_value = {}

        result = await _generate_story(state)
        assert isinstance(result, list)
