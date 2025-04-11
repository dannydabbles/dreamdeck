import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import uuid

import src.oracle_workflow as oracle_workflow_mod
from src.models import ChatState, HumanMessage, AIMessage

@pytest.mark.asyncio
async def test_simple_turn_tool_then_persona(monkeypatch):
    """
    Oracle: roll -> storyteller_gm -> END_TURN
    """
    # Setup dummy state
    state = ChatState(
        messages=[HumanMessage(content="Roll a d20", name="Player")],
        thread_id="thread1",
        current_persona="storyteller_gm",
    )

    # Patch persona classifier to always return storyteller_gm
    monkeypatch.setattr(
        "src.agents.persona_classifier_agent._classify_persona",
        AsyncMock(return_value={"persona": "storyteller_gm", "reason": "test"}),
    )

    # Patch oracle_agent to return "roll" then "storyteller_gm" then "END_TURN"
    oracle_decisions = ["roll", "storyteller_gm", "END_TURN"]
    async def fake_oracle_agent(state, **kwargs):
        return oracle_decisions.pop(0)
    monkeypatch.setattr("src.agents.oracle_agent.oracle_agent", fake_oracle_agent)

    # Patch agents_map: roll returns a dice AIMessage, storyteller_gm returns a GM AIMessage
    fake_dice_msg = AIMessage(content="You rolled a 20!", name="dice_roll", metadata={"message_id": "dice1"})
    fake_gm_msg = AIMessage(content="The dragon appears!", name="Game Master", metadata={"message_id": "gm1"})
    agents_map = {
        "roll": AsyncMock(return_value=[fake_dice_msg]),
        "storyteller_gm": AsyncMock(return_value=[fake_gm_msg]),
    }
    monkeypatch.setattr(oracle_workflow_mod, "agents_map", agents_map)
    monkeypatch.setattr(oracle_workflow_mod, "persona_workflows", {"storyteller_gm": agents_map["storyteller_gm"]})

    # Run workflow
    result_state = await oracle_workflow_mod.oracle_workflow({"messages": state.messages, "previous": state}, state)
    # Should have both dice and GM messages
    assert any(m.content == "You rolled a 20!" for m in result_state.messages)
    assert any(m.content == "The dragon appears!" for m in result_state.messages)
    # tool_results_this_turn should only have the dice message
    assert any(m.content == "You rolled a 20!" for m in result_state.tool_results_this_turn)
    assert not any(m.content == "The dragon appears!" for m in result_state.tool_results_this_turn)

@pytest.mark.asyncio
async def test_multi_tool_turn(monkeypatch):
    """
    Oracle: search -> roll -> storyteller_gm -> END_TURN
    """
    state = ChatState(
        messages=[HumanMessage(content="Search for dragons and roll a d6", name="Player")],
        thread_id="thread2",
        current_persona="storyteller_gm",
    )

    monkeypatch.setattr(
        "src.agents.persona_classifier_agent._classify_persona",
        AsyncMock(return_value={"persona": "storyteller_gm", "reason": "test"}),
    )

    oracle_decisions = ["search", "roll", "storyteller_gm", "END_TURN"]
    async def fake_oracle_agent(state, **kwargs):
        return oracle_decisions.pop(0)
    monkeypatch.setattr("src.agents.oracle_agent.oracle_agent", fake_oracle_agent)

    fake_search_msg = AIMessage(content="Found info on dragons.", name="web_search", metadata={"message_id": "search1"})
    fake_dice_msg = AIMessage(content="You rolled a 6!", name="dice_roll", metadata={"message_id": "dice2"})
    fake_gm_msg = AIMessage(content="A dragon swoops in!", name="Game Master", metadata={"message_id": "gm2"})
    agents_map = {
        "search": AsyncMock(return_value=[fake_search_msg]),
        "roll": AsyncMock(return_value=[fake_dice_msg]),
        "storyteller_gm": AsyncMock(return_value=[fake_gm_msg]),
    }
    monkeypatch.setattr(oracle_workflow_mod, "agents_map", agents_map)
    monkeypatch.setattr(oracle_workflow_mod, "persona_workflows", {"storyteller_gm": agents_map["storyteller_gm"]})

    result_state = await oracle_workflow_mod.oracle_workflow({"messages": state.messages, "previous": state}, state)
    # All three messages should be present
    assert any(m.content == "Found info on dragons." for m in result_state.messages)
    assert any(m.content == "You rolled a 6!" for m in result_state.messages)
    assert any(m.content == "A dragon swoops in!" for m in result_state.messages)
    # tool_results_this_turn should have search and dice, not GM
    assert any(m.content == "Found info on dragons." for m in result_state.tool_results_this_turn)
    assert any(m.content == "You rolled a 6!" for m in result_state.tool_results_this_turn)
    assert not any(m.content == "A dragon swoops in!" for m in result_state.tool_results_this_turn)

@pytest.mark.asyncio
async def test_direct_persona_turn(monkeypatch):
    """
    Oracle: storyteller_gm -> END_TURN
    """
    state = ChatState(
        messages=[HumanMessage(content="Tell me a story", name="Player")],
        thread_id="thread3",
        current_persona="storyteller_gm",
    )

    monkeypatch.setattr(
        "src.agents.persona_classifier_agent._classify_persona",
        AsyncMock(return_value={"persona": "storyteller_gm", "reason": "test"}),
    )

    oracle_decisions = ["storyteller_gm", "END_TURN"]
    async def fake_oracle_agent(state, **kwargs):
        return oracle_decisions.pop(0)
    monkeypatch.setattr("src.agents.oracle_agent.oracle_agent", fake_oracle_agent)

    fake_gm_msg = AIMessage(content="Once upon a time...", name="Game Master", metadata={"message_id": "gm3"})
    agents_map = {
        "storyteller_gm": AsyncMock(return_value=[fake_gm_msg]),
    }
    monkeypatch.setattr(oracle_workflow_mod, "agents_map", agents_map)
    monkeypatch.setattr(oracle_workflow_mod, "persona_workflows", {"storyteller_gm": agents_map["storyteller_gm"]})

    result_state = await oracle_workflow_mod.oracle_workflow({"messages": state.messages, "previous": state}, state)
    assert any(m.content == "Once upon a time..." for m in result_state.messages)
    assert not result_state.tool_results_this_turn  # Should be empty

@pytest.mark.asyncio
async def test_max_iterations_hit(monkeypatch):
    """
    Oracle: returns 'roll' repeatedly, exceeding MAX_CHAIN_LENGTH
    """
    state = ChatState(
        messages=[HumanMessage(content="Keep rolling", name="Player")],
        thread_id="thread4",
        current_persona="storyteller_gm",
    )

    monkeypatch.setattr(
        "src.agents.persona_classifier_agent._classify_persona",
        AsyncMock(return_value={"persona": "storyteller_gm", "reason": "test"}),
    )

    # Always return 'roll'
    async def fake_oracle_agent(state, **kwargs):
        return "roll"
    monkeypatch.setattr("src.agents.oracle_agent.oracle_agent", fake_oracle_agent)

    fake_dice_msg = AIMessage(content="You rolled a 1!", name="dice_roll", metadata={"message_id": "dice3"})
    agents_map = {
        "roll": AsyncMock(return_value=[fake_dice_msg]),
    }
    monkeypatch.setattr(oracle_workflow_mod, "agents_map", agents_map)
    monkeypatch.setattr(oracle_workflow_mod, "persona_workflows", {})

    # Patch MAX_CHAIN_LENGTH to 3 for test
    monkeypatch.setattr(oracle_workflow_mod, "MAX_CHAIN_LENGTH", 3)

    result_state = await oracle_workflow_mod.oracle_workflow({"messages": state.messages, "previous": state}, state)
    # Should have 3 dice messages and a max iteration message
    dice_msgs = [m for m in result_state.messages if m.content == "You rolled a 1!"]
    assert len(dice_msgs) == 3
    assert any("maximum processing steps" in m.content for m in result_state.messages)

@pytest.mark.asyncio
async def test_persona_switch_flow(monkeypatch):
    """
    Persona classifier suggests a switch, Oracle uses new persona.
    """
    state = ChatState(
        messages=[HumanMessage(content="I need therapy", name="Player")],
        thread_id="thread5",
        current_persona="storyteller_gm",
    )

    # Classifier suggests therapist
    monkeypatch.setattr(
        "src.agents.persona_classifier_agent._classify_persona",
        AsyncMock(return_value={"persona": "therapist", "reason": "test"}),
    )

    oracle_decisions = ["therapist", "END_TURN"]
    async def fake_oracle_agent(state, **kwargs):
        return oracle_decisions.pop(0)
    monkeypatch.setattr("src.agents.oracle_agent.oracle_agent", fake_oracle_agent)

    fake_therapist_msg = AIMessage(content="Let's talk about your feelings.", name="Therapist", metadata={"message_id": "therapist1"})
    agents_map = {
        "therapist": AsyncMock(return_value=[fake_therapist_msg]),
    }
    monkeypatch.setattr(oracle_workflow_mod, "agents_map", agents_map)
    monkeypatch.setattr(oracle_workflow_mod, "persona_workflows", {"therapist": agents_map["therapist"]})

    result_state = await oracle_workflow_mod.oracle_workflow({"messages": state.messages, "previous": state}, state)
    assert any(m.content == "Let's talk about your feelings." for m in result_state.messages)
    assert result_state.current_persona == "therapist"

@pytest.mark.asyncio
async def test_tool_agent_error(monkeypatch):
    """
    Tool agent raises exception, should add error message and break.
    """
    state = ChatState(
        messages=[HumanMessage(content="Roll a d20", name="Player")],
        thread_id="thread6",
        current_persona="storyteller_gm",
    )

    monkeypatch.setattr(
        "src.agents.persona_classifier_agent._classify_persona",
        AsyncMock(return_value={"persona": "storyteller_gm", "reason": "test"}),
    )

    oracle_decisions = ["roll"]
    async def fake_oracle_agent(state, **kwargs):
        return oracle_decisions.pop(0) if oracle_decisions else "END_TURN"
    monkeypatch.setattr("src.agents.oracle_agent.oracle_agent", fake_oracle_agent)

    async def error_agent(state, **kwargs):
        raise RuntimeError("Dice broke!")
    agents_map = {
        "roll": error_agent,
    }
    monkeypatch.setattr(oracle_workflow_mod, "agents_map", agents_map)
    monkeypatch.setattr(oracle_workflow_mod, "persona_workflows", {})

    result_state = await oracle_workflow_mod.oracle_workflow({"messages": state.messages, "previous": state}, state)
    assert any("An error occurred while running 'roll'" in m.content for m in result_state.messages)

@pytest.mark.asyncio
async def test_oracle_agent_error(monkeypatch):
    """
    Oracle agent raises exception, should END_TURN and add error message.
    """
    state = ChatState(
        messages=[HumanMessage(content="Tell me a story", name="Player")],
        thread_id="thread7",
        current_persona="storyteller_gm",
    )

    monkeypatch.setattr(
        "src.agents.persona_classifier_agent._classify_persona",
        AsyncMock(return_value={"persona": "storyteller_gm", "reason": "test"}),
    )

    async def error_oracle_agent(state, **kwargs):
        raise RuntimeError("Oracle failed!")
    monkeypatch.setattr("src.agents.oracle_agent.oracle_agent", error_oracle_agent)

    agents_map = {}
    monkeypatch.setattr(oracle_workflow_mod, "agents_map", agents_map)
    monkeypatch.setattr(oracle_workflow_mod, "persona_workflows", {})

    result_state = await oracle_workflow_mod.oracle_workflow({"messages": state.messages, "previous": state}, state)
    # Should add error message and not crash
    assert any("An error occurred in the oracle workflow." in m.content for m in result_state.messages)

# NOTE: The following tests require the initial_chat_state and mock_cl_environment_for_oracle fixtures,
# which are not present in this file. These tests are not runnable as-is and should be removed or refactored.
# See the top of this file for the main integration tests.

# @pytest.mark.asyncio
# async def test_oracle_single_step_persona_agent(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
#     ...

# @pytest.mark.asyncio
# async def test_oracle_multi_step_tool_then_persona(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
#     ...

# @pytest.mark.asyncio
# async def test_oracle_max_iterations_reached(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
#     ...

# @pytest.mark.asyncio
# async def test_oracle_persona_classification_updates_state(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
#     ...

# The following tests require fixtures that are not present in this file and should be removed or refactored.
# They are commented out to prevent pytest from trying to collect them.

# @pytest.mark.asyncio
# async def test_oracle_multi_step_tool_then_persona(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
#     ...

# @pytest.mark.asyncio
# async def test_oracle_max_iterations_reached(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
#     ...

# @pytest.mark.asyncio
# async def test_oracle_persona_classification_updates_state(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
#     ...
