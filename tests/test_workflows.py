import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.workflows import _chat_workflow
from langgraph.func import task
from src.models import ChatState

@pytest.fixture
def mock_chat_state():
    from src.models import ChatState
    return ChatState(messages=[], thread_id="test-thread-id")

@pytest.mark.asyncio
async def test_chat_workflow_memory_updates(mock_chat_state):
    dummy_ai_msg1 = AIMessage(content="Tool1 done", name="tool1", metadata={"message_id": "m1"})
    dummy_knowledge_msg = AIMessage(content="Character info", name="character", metadata={"message_id": "k1"})
    dummy_gm_msg = AIMessage(content="Story continues", name="Game Master", metadata={"message_id": "gm1"})

    with patch("src.workflows.director_agent", new_callable=AsyncMock) as mock_director, \
         patch("src.workflows.agents_map", {"tool1": AsyncMock(return_value=[dummy_ai_msg1])}), \
         patch("src.workflows.knowledge_agent", new_callable=AsyncMock, return_value=[dummy_knowledge_msg]) as mock_knowledge_agent, \
         patch("src.workflows.writer_agent", new_callable=AsyncMock, return_value=[dummy_gm_msg]), \
         patch("src.workflows.cl.user_session.get", new_callable=MagicMock) as mock_user_session_get:

        vector_store = AsyncMock()
        mock_user_session_get.return_value = vector_store

        initial_state = mock_chat_state
        initial_state.messages.append(HumanMessage(content="Hi"))

        mock_director.return_value = ["tool1", {"action": "knowledge", "type": "character"}, "write"]

        updated_state = await _chat_workflow([], previous=initial_state)

        assert dummy_ai_msg1 in updated_state.messages
        assert dummy_knowledge_msg in updated_state.messages
        assert dummy_gm_msg in updated_state.messages

        calls = [call.kwargs for call in vector_store.put.await_args_list]
        msg_ids = [c["message_id"] for c in calls]
        assert "m1" in msg_ids
        assert "k1" in msg_ids
        assert "gm1" in msg_ids


@pytest.mark.asyncio
async def test_storyboard_triggered_after_gm(monkeypatch):
    from src.workflows import _chat_workflow

    dummy_gm_msg = AIMessage(content="Story continues", name="Game Master", metadata={"message_id": "gm123"})
    dummy_state = ChatState(messages=[HumanMessage(content="Hi")], thread_id="t1")

    with patch("src.workflows.director_agent", new_callable=AsyncMock) as mock_director, \
         patch("src.workflows.writer_agent", new_callable=AsyncMock, return_value=[dummy_gm_msg]) as mock_writer, \
         patch("src.workflows.cl.user_session.get", return_value=None), \
         patch("src.workflows.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard:

        mock_director.return_value = ["write"]

        updated_state = await _chat_workflow([], previous=dummy_state)

        mock_storyboard.assert_called_once()
        args, kwargs = mock_storyboard.call_args
        assert kwargs["state"] == updated_state
        assert kwargs["gm_message_id"] == "gm123"


@pytest.mark.asyncio
async def test_multi_hop_orchestration(monkeypatch):
    from src.workflows import _chat_workflow

    dummy_search = AIMessage(content="Search results", name="search", metadata={"message_id": "s1"})
    dummy_knowledge = AIMessage(content="Lore details", name="lore", metadata={"message_id": "k1"})
    dummy_gm = AIMessage(content="Story", name="Game Master", metadata={"message_id": "gm1"})

    state = ChatState(messages=[HumanMessage(content="Hi")], thread_id="t1")

    with patch("src.workflows.director_agent", new_callable=AsyncMock) as mock_director, \
         patch("src.workflows.knowledge_agent", new_callable=AsyncMock, return_value=[dummy_knowledge]) as mock_knowledge, \
         patch("src.workflows.agents_map", {
             "search": AsyncMock(return_value=[dummy_search]),
         }), \
         patch("src.workflows.writer_agent", new_callable=AsyncMock, return_value=[dummy_gm]), \
         patch("src.workflows.cl.user_session.get", return_value=None):

        mock_director.side_effect = [
            ["search"],
            [{"action": "knowledge", "type": "lore"}],
            ["write"]
        ]

        updated_state = await _chat_workflow([], previous=state)

        contents = [m.content for m in updated_state.messages]
        assert "Search results" in contents
        assert "Lore details" in contents
        assert "Story" in contents


@pytest.mark.asyncio
async def test_workflow_error_fallback(monkeypatch):
    from src.workflows import _chat_workflow

    state = ChatState(messages=[HumanMessage(content="Hi")], thread_id="t1")

    with patch("src.workflows.director_agent", new_callable=AsyncMock, side_effect=Exception("fail")), \
         patch("src.workflows.cl.user_session.get", return_value=None):

        updated_state = await _chat_workflow([], previous=state)

        assert updated_state.messages[-1].content == "The adventure continues..."
