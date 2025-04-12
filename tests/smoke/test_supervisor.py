import pytest
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage
from unittest.mock import AsyncMock, patch

import sys

@pytest.mark.asyncio
async def test_supervisor_tool_routing(monkeypatch):
    # Patch get_agent to return the undecorated function to avoid langgraph context issues
    from src.agents.dice_agent import _dice_roll
    dummy_agent = AsyncMock(side_effect=_dice_roll)
    with patch("src.agents.registry.get_agent", return_value=dummy_agent), patch("src.supervisor.task", lambda x: x):
        from src.supervisor import supervisor
        state = ChatState(messages=[HumanMessage(content="/dice", name="Player")], thread_id="t1")
        result = await supervisor(state)
        # Accept either the dummy_agent's return or the real _dice_roll's output
        assert isinstance(result, list)
        assert result and isinstance(result[0], AIMessage)

@pytest.mark.asyncio
async def test_supervisor_storyboard_routing(monkeypatch):
    # Patch get_agent to return the undecorated function to avoid langgraph context issues
    from src.agents.storyboard_editor_agent import _generate_storyboard
    dummy_agent = AsyncMock(side_effect=lambda state, gm_message_id=None: _generate_storyboard(state, gm_message_id or "gm1"))
    with patch("src.agents.registry.get_agent", return_value=dummy_agent), patch("src.supervisor.task", lambda x: x):
        from src.supervisor import supervisor
        # Add a GM message with message_id
        gm_msg = AIMessage(content="scene", name="Game Master", metadata={"message_id": "gm1"})
        state = ChatState(messages=[HumanMessage(content="/storyboard", name="Player"), gm_msg], thread_id="t1")
        result = await supervisor(state)
        assert isinstance(result, list)
        assert result and isinstance(result[0], AIMessage)

@pytest.mark.asyncio
async def test_supervisor_persona_routing(monkeypatch):
    # Patch writer_agent.persona_agent_registry and get_agent to avoid langgraph context issues
    dummy_agent = AsyncMock(return_value=[AIMessage(content="persona result", name="writer")])
    with patch("src.agents.writer_agent.persona_agent_registry", {"default": dummy_agent}), \
         patch("src.agents.registry.get_agent", return_value=None), \
         patch("src.supervisor.task", lambda x: x):
        from src.supervisor import supervisor
        state = ChatState(messages=[HumanMessage(content="continue", name="Player")], thread_id="t1", current_persona="default")
        result = await supervisor(state)
        assert result[0].content == "persona result"
