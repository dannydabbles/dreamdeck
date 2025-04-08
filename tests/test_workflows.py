import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.workflows import _chat_workflow
from src.agents.decision_agent import decide_action
from langgraph.func import task  # Ensure proper imports
from src.models import ChatState  # <-- Add this import

@pytest.fixture
def mock_chat_state():
    from src.models import ChatState
    return ChatState(messages=[], thread_id="test-thread-id")

@pytest.mark.asyncio
async def test_chat_workflow_memory_updates(mock_chat_state):
    dummy_ai_msg1 = AIMessage(content="Tool1 done", name="tool1", metadata={"message_id": "m1"})
    dummy_ai_msg2 = AIMessage(content="Tool2 done", name="tool2", metadata={"message_id": "m2"})
    dummy_gm_msg = AIMessage(content="Story continues", name="Game Master", metadata={"message_id": "gm1"})

    with patch("src.workflows.orchestrator_agent", new_callable=AsyncMock) as mock_orchestrator, \
         patch("src.workflows.agents_map", {"tool1": AsyncMock(return_value=[dummy_ai_msg1]), "tool2": AsyncMock(return_value=[dummy_ai_msg2])}), \
         patch("src.workflows.writer_agent", new_callable=AsyncMock, return_value=[dummy_gm_msg]), \
         patch("src.workflows.cl.user_session.get", new_callable=MagicMock) as mock_user_session_get:

        vector_store = AsyncMock()
        mock_user_session_get.return_value = vector_store

        initial_state = mock_chat_state
        initial_state.messages.append(HumanMessage(content="Hi"))

        mock_orchestrator.return_value = ["tool1", "tool2"]

        updated_state = await _chat_workflow([], previous=initial_state)

        # All AI messages appended
        assert dummy_ai_msg1 in updated_state.messages
        assert dummy_ai_msg2 in updated_state.messages
        assert dummy_gm_msg in updated_state.messages

        # Vector store put called for each AI message
        calls = [call.kwargs for call in vector_store.put.await_args_list]
        msg_ids = [c["message_id"] for c in calls]
        assert "m1" in msg_ids
        assert "m2" in msg_ids
        assert "gm1" in msg_ids


@pytest.mark.asyncio
async def test_storyboard_triggered_after_gm(monkeypatch):
    from src.workflows import _chat_workflow

    dummy_gm_msg = AIMessage(content="Story continues", name="Game Master", metadata={"message_id": "gm123"})
    dummy_state = ChatState(messages=[HumanMessage(content="Hi")], thread_id="t1")

    # Patch orchestrator to return no tools, so GM is called immediately
    with patch("src.workflows.orchestrator_agent", new_callable=AsyncMock) as mock_orch, \
         patch("src.workflows.writer_agent", new_callable=AsyncMock, return_value=[dummy_gm_msg]) as mock_writer, \
         patch("src.workflows.cl.user_session.get", return_value=None), \
         patch("src.workflows.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard:

        mock_orch.return_value = ["write"]

        updated_state = await _chat_workflow([], previous=dummy_state)

        # Because storyboard_editor_agent is called but *not awaited* in _chat_workflow,
        # it is a coroutine that is never awaited, so AsyncMock.await_count remains 0.
        # Instead, check it was *called* (sync), not awaited.
        mock_storyboard.assert_called_once()
        args, kwargs = mock_storyboard.call_args
        assert kwargs["state"] == updated_state
        assert kwargs["gm_message_id"] == "gm123"


@pytest.mark.asyncio
async def test_multi_hop_orchestration(monkeypatch):
    from src.workflows import _chat_workflow

    dummy_tool1 = AIMessage(content="Tool1 done", name="tool1", metadata={"message_id": "m1"})
    dummy_tool2 = AIMessage(content="Tool2 done", name="tool2", metadata={"message_id": "m2"})
    dummy_gm = AIMessage(content="Story", name="Game Master", metadata={"message_id": "gm1"})

    state = ChatState(messages=[HumanMessage(content="Hi")], thread_id="t1")

    with patch("src.workflows.orchestrator_agent", new_callable=AsyncMock) as mock_orch, \
         patch("src.workflows.agents_map", {
             "tool1": AsyncMock(return_value=[dummy_tool1]),
             "tool2": AsyncMock(return_value=[dummy_tool2]),
         }), \
         patch("src.workflows.writer_agent", new_callable=AsyncMock, return_value=[dummy_gm]), \
         patch("src.workflows.cl.user_session.get", return_value=None):

        # First orchestrator call returns tool1
        # Second returns tool2
        # Third returns write (GM)
        mock_orch.side_effect = [
            ["tool1"],
            ["tool2"],
            ["write"]
        ]

        updated_state = await _chat_workflow([], previous=state)

        # All tool and GM messages appended
        contents = [m.content for m in updated_state.messages]
        assert "Tool1 done" in contents
        assert "Tool2 done" in contents
        assert "Story" in contents


@pytest.mark.asyncio
async def test_workflow_error_fallback(monkeypatch):
    from src.workflows import _chat_workflow

    state = ChatState(messages=[HumanMessage(content="Hi")], thread_id="t1")

    with patch("src.workflows.orchestrator_agent", new_callable=AsyncMock, side_effect=Exception("fail")), \
         patch("src.workflows.cl.user_session.get", return_value=None):

        updated_state = await _chat_workflow([], previous=state)

        # Last message should be fallback
        assert updated_state.messages[-1].content == "The adventure continues..."
