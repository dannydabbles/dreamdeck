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
async def test_chat_workflow(mock_chat_state):
    from src.agents.decision_agent import decide_action
    from langgraph.func import task
    from unittest.mock import MagicMock

    with (
        patch("langgraph.func.task", new=lambda f: f),
        patch("langchain_openai.ChatOpenAI.ainvoke") as mock_llm_invoke,
        patch("chainlit.user_session") as mock_cl_session  # Removed 'src.'
    ):
        mock_llm_invoke.return_value = MagicMock(content="continue_story")
        mock_cl_session.get.return_value = MagicMock()  # Mock vector store

        initial_state = mock_chat_state
        new_messages = [HumanMessage(content="Test input")]

        updated_state = await _chat_workflow(new_messages, previous=initial_state)

        assert updated_state.messages[-1].content == "The adventure continues..."
