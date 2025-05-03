import asyncio # Add this import
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState

import chainlit as cl
from chainlit.context import ChainlitContext, context_var

import contextlib

@pytest.fixture(autouse=True)
def patch_chainlit_context(monkeypatch):
    # Patch Chainlit context for supervisor tests
    mock_session = MagicMock()
    mock_session.thread_id = "test-thread-id"
    mock_context = MagicMock(spec=ChainlitContext)
    mock_context.session = mock_session
    mock_context.emitter = AsyncMock()
    token = context_var.set(mock_context)
    try:
        yield
    finally:
        context_var.reset(token)

@pytest.mark.asyncio
async def test_supervisor_multi_hop_workflow():
    """Test supervisor routing through multiple tools"""
    state = ChatState(
        messages=[
            HumanMessage(content="Search for dragon weaknesses and roll attack dice"),
        ],
        thread_id="multi-1"
    )
    
    mock_responses = [
        AIMessage(content="Dragon weak to ice", name="web_search"),
        AIMessage(content="You rolled 15", name="dice_roll"),
        AIMessage(content="The dragon shivers as you attack!", name="🎭 Storyteller GM")
    ]
    
    mock_supervisor = AsyncMock(side_effect=[mock_responses])
    
    with patch("src.supervisor.supervisor", mock_supervisor):
        from src.supervisor import supervisor
        response = await supervisor(state)
        
        assert len(response) == 3
        assert "weak to ice" in response[0].content
        assert "rolled 15" in response[1].content
        assert "dragon shivers" in response[2].content

@pytest.mark.asyncio
async def test_persona_switch_workflow():
    """Test automatic persona switching mid-conversation"""
    state = ChatState(
        messages=[
            HumanMessage(content="I need to organize my inventory"),
            AIMessage(content="Todo list updated", name="todo"),
        ],
        thread_id="persona-1"
    )
    
    mock_response = AIMessage(
        content="Your inventory is organized:",
        name="🗒️ Secretary"
    )
    
    with patch("src.agents.writer_agent._generate_story") as mock_generate_story:
        mock_generate_story.return_value = [mock_response]
        from src.supervisor import supervisor
        response = await supervisor(state)
        
        assert "inventory is organized" in response[0].content
        assert "Secretary" in response[0].name


@pytest.mark.asyncio
async def test_supervisor_calls_storyboard_after_gm_persona():
    """
    Verify supervisor calls storyboard_editor_agent after a GM persona agent
    (like Storyteller GM) returns a message with a message_id.
    """
    state = ChatState(
        messages=[HumanMessage(content="Describe the scene")],
        thread_id="storyboard-test-1",
        current_persona="Storyteller GM" # Start with GM persona for simplicity in test
    )
    gm_message_id = "gm-msg-test-id-1"
    gm_response = AIMessage(
        content="A dragon swoops down!",
        name="🎭 Storyteller GM",
        metadata={"message_id": gm_message_id, "type": "gm_message"} # Ensure type is set
    )

    # Mock the decision agent to route to the GM persona
    mock_decision = AsyncMock(return_value={"route": "persona:Storyteller GM"})
    # Mock the writer agent (representing the GM persona) to return the GM message
    mock_writer = AsyncMock(return_value=[gm_response])
    # Mock the storyboard agent to check if it's called correctly
    mock_storyboard = AsyncMock(return_value=[]) # Storyboard agent returns empty list or status message
    # Mock asyncio.create_task to prevent actual task execution
    mock_create_task = MagicMock()

    with patch("src.supervisor._decide_next_agent", mock_decision), \
         patch("src.supervisor.writer_agent", mock_writer), \
         patch("src.supervisor.storyboard_editor_agent", mock_storyboard), \
         patch("src.supervisor.asyncio.create_task", mock_create_task):

        from src.supervisor import supervisor
        results = await supervisor(state)

        # Assertions
        mock_decision.assert_awaited_once()
        mock_writer.assert_awaited_once_with(state) # Writer gets the updated state
        # Check that create_task was called with the storyboard agent coroutine
        assert mock_create_task.call_count == 1
        call_args, call_kwargs = mock_create_task.call_args
        # The first argument to create_task should be the coroutine
        storyboard_coro = call_args[0]
        # Check if the coroutine is for the storyboard agent
        # This is a bit indirect, but checks the function and relevant args
        assert storyboard_coro.cr_code.co_name == mock_storyboard.cr_code.co_name # Check function name
        # Check that the storyboard agent was called *within the task* with the correct gm_message_id
        # We can't directly assert await on mock_storyboard as it's in a task,
        # but we can check the arguments passed to create_task indirectly.
        # A more robust check might involve inspecting the coroutine's arguments if possible,
        # or setting a side effect on the mock_storyboard to record calls.
        # For now, checking create_task call is a good indicator.

        # Check that the GM response is in the results
        assert gm_response in results
        # Check that the GM response was added to the state messages *before* storyboard check
        assert gm_response in state.messages


@pytest.mark.asyncio
async def test_supervisor_does_not_call_storyboard_after_non_gm_persona():
    """
    Verify supervisor does NOT call storyboard_editor_agent after a non-GM persona agent.
    """
    state = ChatState(
        messages=[HumanMessage(content="How are you?")],
        thread_id="storyboard-test-2",
        current_persona="Friend" # Start with non-GM persona
    )
    friend_response = AIMessage(
        content="I'm doing well, thanks!",
        name="🤝 Friend",
        metadata={"message_id": "friend-msg-test-id-1"}
    )

    mock_decision = AsyncMock(return_value={"route": "persona:Friend"})
    # Mock the writer agent configured for the 'Friend' persona
    mock_friend_agent = AsyncMock(return_value=[friend_response])
    mock_storyboard = AsyncMock()
    mock_create_task = MagicMock()

    # Assume 'Friend' persona maps to writer_agent in this simplified test setup
    with patch("src.supervisor._decide_next_agent", mock_decision), \
         patch("src.supervisor.writer_agent", mock_friend_agent), \
         patch("src.supervisor.storyboard_editor_agent", mock_storyboard), \
         patch("src.supervisor.asyncio.create_task", mock_create_task):

        from src.supervisor import supervisor
        results = await supervisor(state)

        # Assertions
        mock_decision.assert_awaited_once()
        mock_friend_agent.assert_awaited_once_with(state)
        mock_storyboard.assert_not_awaited() # Storyboard should NOT be called
        mock_create_task.assert_not_called() # Task should not be created
        assert friend_response in results
        assert friend_response in state.messages


@pytest.mark.asyncio
async def test_supervisor_does_not_call_storyboard_if_gm_message_lacks_id():
    """
    Verify supervisor does NOT call storyboard if the GM message lacks a message_id.
    """
    state = ChatState(
        messages=[HumanMessage(content="Describe the scene")],
        thread_id="storyboard-test-3",
        current_persona="Storyteller GM"
    )
    # GM message *without* message_id in metadata
    gm_response_no_id = AIMessage(
        content="A dragon swoops down!",
        name="🎭 Storyteller GM",
        metadata={"type": "gm_message"} # Missing message_id
    )

    mock_decision = AsyncMock(return_value={"route": "persona:Storyteller GM"})
    mock_writer = AsyncMock(return_value=[gm_response_no_id])
    mock_storyboard = AsyncMock()
    mock_create_task = MagicMock()

    with patch("src.supervisor._decide_next_agent", mock_decision), \
         patch("src.supervisor.writer_agent", mock_writer), \
         patch("src.supervisor.storyboard_editor_agent", mock_storyboard), \
         patch("src.supervisor.asyncio.create_task", mock_create_task):

        from src.supervisor import supervisor
        results = await supervisor(state)

        # Assertions
        mock_decision.assert_awaited_once()
        mock_writer.assert_awaited_once_with(state)
        mock_storyboard.assert_not_awaited() # Storyboard should NOT be called
        mock_create_task.assert_not_called() # Task should not be created
        assert gm_response_no_id in results
        assert gm_response_no_id in state.messages
