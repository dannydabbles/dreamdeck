import pytest
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage
from unittest.mock import AsyncMock, patch

import sys

@pytest.mark.asyncio
async def test_supervisor_tool_routing(monkeypatch):
    # Patch get_agent to return a dummy agent
    dummy_agent = AsyncMock(return_value=[AIMessage(content="tool result", name="dice")])
    with patch("src.agents.registry.get_agent", return_value=dummy_agent), patch("src.supervisor.task", lambda x: x):
        from src.supervisor import supervisor
        state = ChatState(messages=[HumanMessage(content="/dice", name="Player")], thread_id="t1")
        result = await supervisor(state)
        assert result[0].content == "tool result"

@pytest.mark.asyncio
async def test_supervisor_storyboard_routing(monkeypatch):
    # Patch get_agent to return a dummy storyboard agent
    dummy_agent = AsyncMock(return_value=[AIMessage(content="storyboard", name="storyboard")])
    with patch("src.agents.registry.get_agent", return_value=dummy_agent), patch("src.supervisor.task", lambda x: x):
        from src.supervisor import supervisor
        # Add a GM message with message_id
        gm_msg = AIMessage(content="scene", name="Game Master", metadata={"message_id": "gm1"})
        state = ChatState(messages=[HumanMessage(content="/storyboard", name="Player"), gm_msg], thread_id="t1")
        result = await supervisor(state)
        assert result[0].content == "storyboard"

@pytest.mark.asyncio
async def test_supervisor_persona_routing(monkeypatch):
    # Patch writer_agent.persona_agent_registry to return a dummy agent
    dummy_agent = AsyncMock(return_value=[AIMessage(content="persona result", name="writer")])
    with patch("src.agents.writer_agent.persona_agent_registry", {"default": dummy_agent}), patch("src.supervisor.task", lambda x: x):
        from src.supervisor import supervisor
        state = ChatState(messages=[HumanMessage(content="continue", name="Player")], thread_id="t1", current_persona="default")
        result = await supervisor(state)
        assert result[0].content == "persona result"
