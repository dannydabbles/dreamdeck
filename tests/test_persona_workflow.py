import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState
import src.workflows as workflows_module

@pytest.mark.asyncio
async def test_persona_workflow_filters_and_reorders(monkeypatch):
    # Explicitly override PERSONA_TOOL_PREFERENCES inside the test
    workflows_module.PERSONA_TOOL_PREFERENCES = {
        "therapist": {
            "avoid": ["roll"],
            "prefer": ["knowledge"]
        },
        "default": {}
    }

    # Prepare dummy previous state
    state = ChatState(
        messages=[
            HumanMessage(content="Hello", name="Player", metadata={})
        ],
        thread_id="test_thread",
        current_persona="Therapist",  # Persona with avoid=["roll"], prefer=["knowledge"]
    )

    # Mock director_agent to return mixed actions
    async def fake_director(state_arg):
        # Return a mix of strings and dicts, including "roll" and "knowledge"
        return ["roll", {"action": "knowledge", "type": "lore"}, "todo", "write"]

    # Patch director_agent in workflows module
    monkeypatch.setattr(workflows_module, "director_agent", fake_director)

    # Patch writer_agent to just return a dummy AIMessage
    async def fake_writer(state_arg):
        return [AIMessage(content="Story continues", name="Game Master", metadata={"message_id": "gm123"})]

    monkeypatch.setattr(workflows_module, "writer_agent", fake_writer)

    # Patch knowledge_agent to return dummy AIMessage
    async def fake_knowledge(state_arg, knowledge_type=None, **kwargs):
        return [AIMessage(content=f"Knowledge: {knowledge_type}", name="knowledge", metadata={"message_id": f"kn_{knowledge_type}"})]

    monkeypatch.setattr(workflows_module, "knowledge_agent", fake_knowledge)

    # Patch storyboard_editor_agent to do nothing
    async def fake_storyboard(state_arg, gm_message_id=None):
        return []

    monkeypatch.setattr(workflows_module, "storyboard_editor_agent", fake_storyboard)

    # Patch cl.user_session.get to avoid errors
    def fake_user_session_get(key, default=None):
        if key == "config":
            return {"configurable": {}}
        return {}
    monkeypatch.setattr("chainlit.user_session.get", fake_user_session_get)

    # Patch the entire agents_map to avoid calling real dice agent etc.
    monkeypatch.setattr(workflows_module, "agents_map", {
        "roll": lambda *_args, **_kwargs: [],
        "todo": lambda *_args, **_kwargs: [],
        "write": fake_writer,
        "continue_story": fake_writer,
    })

    # Run the workflow
    new_state = await workflows_module._chat_workflow(
        messages=[],
        previous=state,
    )

    # The initial director returns ["roll", {"action": "knowledge", "type": "lore"}, "todo", "write"]
    # Therapist persona avoid=["roll"], prefer=["knowledge"]
    # So after filtering/reordering, actions should be:
    # [{"action": "knowledge", "type": "lore"}, "todo", "write"]
    # with "knowledge" first (preferred), "roll" removed

    # Check that the first AI message is from knowledge agent
    knowledge_msgs = [m for m in new_state.messages if isinstance(m, AIMessage) and m.name == "knowledge"]
    assert knowledge_msgs, "Knowledge agent message should be present"
    assert "Knowledge: lore" in knowledge_msgs[0].content

    # Check that no dice_roll message is present (since 'roll' was avoided)
    dice_msgs = [m for m in new_state.messages if isinstance(m, AIMessage) and m.name == "dice_roll"]
    assert not dice_msgs, "Dice roll should be skipped for therapist persona"

    # Check that the final message is from writer agent
    writer_msgs = [m for m in new_state.messages if isinstance(m, AIMessage) and m.name == "Game Master"]
    assert writer_msgs, "Writer agent message should be present"
    assert "Story continues" in writer_msgs[-1].content
