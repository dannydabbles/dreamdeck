import pytest
from unittest.mock import patch, AsyncMock
from langchain_core.messages import AIMessage, HumanMessage
from src.workflows import _chat_workflow
from src.agents.decision_agent import decide_action
from langgraph.func import task  # Ensure proper imports

@pytest.mark.asyncio
async def test_chat_workflow(mock_chat_state):
    from src.agents.decision_agent import decide_action
    from langgraph.func import task  # Ensure proper imports

    with (
        patch("src.agents.decision_agent.task", new=lambda f: f), 
        patch("src.langgraph.func.task", new=lambda f: f),  # Patch task decorator
        patch("src.langchain_openai.ChatOpenAI.ainvoke") as mock_llm_invoke,
        patch("src.chainlit.user_session") as mock_cl_session
    ):
        mock_llm_invoke.return_value = MagicMock(content="continue_story")
        mock_cl_session.get.return_value = MagicMock()  # Mock vector store

        initial_state = mock_chat_state
        new_messages = [HumanMessage(content="Test input")]

        updated_state = await _chat_workflow(new_messages, previous=initial_state)

        assert updated_state.messages[-1].content == "The adventure continues..."
