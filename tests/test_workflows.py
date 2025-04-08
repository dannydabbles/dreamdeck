import pytest
from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage
from src.workflows import _chat_workflow
from src.agents.decision_agent import decide_action
from langgraph.func import task  # Ensure proper imports

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
