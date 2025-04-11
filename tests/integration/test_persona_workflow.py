import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState
import src.workflows as workflows_module
import copy
from src.config import PERSONA_TOOL_PREFERENCES as CONFIG_PREFS


@pytest.mark.asyncio
async def test_persona_workflow_filters_and_reorders(monkeypatch):
    # Use the real config persona preferences instead of hardcoded dict
    workflows_module.PERSONA_TOOL_PREFERENCES = copy.deepcopy(CONFIG_PREFS)

    # Prepare dummy previous state
    state = ChatState(
        messages=[HumanMessage(content="Hello", name="Player", metadata={})],
        thread_id="test_thread",
        current_persona="Therapist",  # Persona with avoid=["roll"], prefer=["knowledge"]
    )

    # Mock Oracle agent to simulate the filtered/reordered sequence
    # Therapist avoids roll, prefers knowledge. Input implies: roll, knowledge, todo, write
    # Expected Oracle sequence: knowledge, todo, therapist (persona agent last)
    oracle_call_count = 0
    async def fake_oracle(state, **kwargs):
        nonlocal oracle_call_count
        oracle_call_count += 1
        if oracle_call_count == 1:
            return "knowledge"
        elif oracle_call_count == 2:
            return "todo"
        elif oracle_call_count == 3:
            return "therapist" # Persona agent is last
        else:
            return "END_TURN"

    # Patch Oracle agent in the oracle_workflow module
    monkeypatch.setattr("src.oracle_workflow.oracle_agent", fake_oracle)

    # Patch writer_agent's underlying function (_generate_story) used by persona workflows
    async def fake_writer(state_arg, **kwargs):
        return [
            AIMessage(
                content="Story continues",
                name="Game Master",
                metadata={"message_id": "gm123"},
            )
        ]

    monkeypatch.setattr("src.persona_workflows._generate_story", fake_writer)

    # Patch knowledge_agent function to return dummy AIMessage
    async def fake_knowledge(state_arg, **kwargs): # knowledge_agent takes state
        return [
            AIMessage(
                content="Knowledge result",
                name="knowledge",
                metadata={"message_id": "kn1"},
            )
        ]
    # Patch the knowledge agent function where it's used (likely in agents_map)
    # We'll patch the map below.

    # Patch todo_agent function to return dummy AIMessage
    async def fake_todo(state_arg, **kwargs): # todo_agent takes state
        return [
            AIMessage(
                content="Todo result",
                name="todo",
                metadata={"message_id": "td1"},
            )
        ]
    # Patch the todo agent function where it's used (likely in agents_map)
    # We'll patch the map below.

    # Patch therapist workflow (already uses fake_writer via _generate_story patch)
    # No, we need a specific mock for the therapist workflow itself
    async def fake_therapist_workflow(inputs, state, **kwargs):
         return [AIMessage(content="Therapy response", name="therapist", metadata={"message_id": "w1"})]


    # Patch cl.user_session.get to avoid errors in the workflow
    def fake_user_session_get(key, default=None):
        if key == "state": return state # Return the initial state
        if key == "vector_memory": return MagicMock()
        if key == "chat_settings": return {}
        return default
    monkeypatch.setattr("chainlit.user_session.get", fake_user_session_get)
    monkeypatch.setattr("chainlit.user_session.set", MagicMock()) # Mock set as well

    # Patch the agents_map used by oracle_workflow
    import src.oracle_workflow as owf
    agents_map_patch = owf.agents_map.copy()
    agents_map_patch["knowledge"] = fake_knowledge
    agents_map_patch["todo"] = fake_todo
    agents_map_patch["therapist"] = fake_therapist_workflow
    # Ensure 'roll' is NOT called
    async def fail_roll(state, **kwargs):
         pytest.fail("Dice agent 'roll' should have been avoided by Oracle")
    agents_map_patch["roll"] = fail_roll
    monkeypatch.setattr(owf, "agents_map", agents_map_patch)

    # Patch necessary functions within oracle_workflow itself if needed
    monkeypatch.setattr(owf, "append_log", lambda *a, **kw: None)
    monkeypatch.setattr(owf, "get_persona_daily_dir", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(owf, "save_text_file", lambda *a, **kw: None)
    monkeypatch.setattr(owf, "persona_classifier_agent", AsyncMock(return_value={"persona": "Therapist"})) # Assume classifier runs

    # Run the workflow using the _chat_workflow alias which wraps oracle_workflow
    # Pass state as the second argument
    new_state = await workflows_module._chat_workflow(
        {"messages": state.messages, "previous": state}, # Pass input dict
        state, # Pass state object
        config={"configurable": {"thread_id": state.thread_id}} # Pass config
    )

    # --- Assertions ---
    assert isinstance(new_state, ChatState), f"Expected ChatState, got {type(new_state)}"

    # Check the sequence of AI messages added
    initial_msg_count = len(state.messages)
    ai_msgs = [m for m in new_state.messages[initial_msg_count:] if isinstance(m, AIMessage)]
    names = [m.name for m in ai_msgs]

    # Expected sequence based on fake_oracle: knowledge -> todo -> therapist
    assert names == ["knowledge", "todo", "therapist"], f"Unexpected agent sequence: {names}"
    assert oracle_call_count == 3, f"Expected 3 Oracle calls, got {oracle_call_count}"

    # Check that no dice_roll message is present
    assert "dice_roll" not in names, "Dice roll should be skipped for therapist persona"
