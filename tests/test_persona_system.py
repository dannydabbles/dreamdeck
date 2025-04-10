import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call  # Added MagicMock, call
from src.agents.persona_classifier_agent import _classify_persona, PERSONA_LIST  # Changed import
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage
import chainlit as cl

import pytest

@pytest.fixture
def mock_cl_environment(monkeypatch):
    # Minimal dummy fixture to avoid import error
    pass

from src.event_handlers import on_message
from src.agents.writer_agent import _generate_story, call_writer_agent  # Changed import
from src.agents.director_agent import _direct_actions  # Changed import
from src.workflows import _chat_workflow  # Changed import
from src.agents.dice_agent import _dice_roll  # Added import
from src.agents.todo_agent import _manage_todo  # Added import


@pytest.mark.asyncio
async def test_persona_classifier_returns_valid_persona(monkeypatch, mock_cl_environment):
    dummy_state = ChatState(messages=[
        HumanMessage(content="I want to write some code", name="Player")
    ], thread_id="thread1")

    mock_response = AsyncMock()
    mock_response.content = '{"persona": "coder", "reason": "User mentioned code"}'
    with patch("src.agents.persona_classifier_agent.ChatOpenAI.ainvoke", return_value=mock_response):
        result = await _classify_persona(dummy_state)
        assert result["persona"] in PERSONA_LIST
        assert "reason" in result
        assert result["persona"] == "coder"
        assert "User mentioned code" in result["reason"]


@pytest.mark.asyncio
async def test_persona_switch_confirmation(monkeypatch, mock_cl_environment):
    dummy_state = ChatState(messages=[], thread_id="thread1")
    # cl.user_session.set("state", dummy_state)  # Removed this line
    cl.user_session.set("current_persona", "default")  # Added this line
    cl.user_session.set("pending_persona_switch", "therapist")

    msg_yes = cl.Message(content="Yes", author="Player")
    await on_message(msg_yes)
    assert cl.user_session.get("pending_persona_switch") is None
    assert cl.user_session.get("current_persona") == "therapist"

    cl.user_session.set("pending_persona_switch", "coder")
    msg_no = cl.Message(content="No", author="Player")
    await on_message(msg_no)
    assert cl.user_session.get("pending_persona_switch") is None
    # Assert that persona remains the last accepted one ('therapist' in this case)
    assert cl.user_session.get("current_persona") == "therapist"


@pytest.mark.asyncio
async def test_writer_agent_selects_persona_prompt(monkeypatch, mock_cl_environment):
    dummy_state = ChatState(messages=[], thread_id="thread1", current_persona="Secretary")

    async def fake_stream(*args, **kwargs):
        class FakeChunk:
            content = "Test story content"
        yield FakeChunk()

    class DummyMsg:
        content = ""
        id = "msgid"
        async def stream_token(self, chunk): self.content += chunk
        async def send(self): pass

    # Patch the ChatOpenAI class used within the writer_agent module
    with patch("src.agents.writer_agent.ChatOpenAI") as MockChatOpenAI:
        # Configure the mock instance that will be created inside the agent
        mock_instance = MockChatOpenAI.return_value
        mock_instance.astream = fake_stream

        # Mock cl.Message creation and its methods
        with patch("src.agents.writer_agent.cl.Message", return_value=DummyMsg()) as mock_cl_msg:
            # Call the actual agent function
            result = await call_writer_agent(dummy_state)

            # Assert that ChatOpenAI was called
            MockChatOpenAI.assert_called_once()
            # Assert cl.Message was used correctly
            mock_cl_msg.assert_called_once()
            # Assert the returned message has the correct persona icon/name
            assert result[0].name.startswith("🗒️ Secretary")


@pytest.mark.asyncio
async def test_director_includes_persona(monkeypatch, mock_cl_environment):
    dummy_state = ChatState(messages=[
        HumanMessage(content="Tell me a story", name="Player")
    ], thread_id="thread1", current_persona="Therapist")

    mock_response = AsyncMock()
    mock_response.content = '{"actions": ["continue_story"]}'
    with patch("src.agents.director_agent.ChatOpenAI.ainvoke", return_value=mock_response) as mock_ainvoke:
        actions = await _direct_actions(dummy_state)
        assert isinstance(actions, list)
        assert "continue_story" in actions
        # Check that the prompt passed to LLM includes persona info
        call_args, call_kwargs = mock_ainvoke.call_args
        system_prompt = call_args[0][0][1]
        assert "Current persona: Therapist" in system_prompt
        assert "Persona tool preferences:" in system_prompt


@pytest.mark.asyncio
async def test_workflow_filters_avoided_tools(monkeypatch, mock_cl_environment):
    dummy_state = ChatState(messages=[
        HumanMessage(content="I need therapy", name="Player")
    ], thread_id="thread1", current_persona="therapist")

    async def fake_director(state):
        return ["roll", "write"]

    async def fake_dice(state, **kwargs):
        pytest.fail("Dice agent should have been filtered out")

    async def fake_writer(state, **kwargs):
        return [AIMessage(content="Therapy response", name="Therapist", metadata={"message_id": "w1"})]

    # Patch the underlying functions, not the @task decorated ones
    monkeypatch.setattr("src.agents.director_agent._direct_actions", fake_director)
    monkeypatch.setattr("src.agents.dice_agent._dice_roll", fake_dice)
    monkeypatch.setattr("src.agents.writer_agent._generate_story", fake_writer)

    # Mock vector store get method if needed
    mock_vector_store = MagicMock()
    mock_vector_store.get = MagicMock(return_value=[])
    mock_vector_store.put = AsyncMock()
    cl.user_session.set("vector_memory", mock_vector_store)

    result_state = await _chat_workflow(dummy_state.messages, dummy_state)
    assert any(m.name == "Therapist" for m in result_state.messages)


@pytest.mark.asyncio
async def test_simulated_conversation_flow(monkeypatch, mock_cl_environment):
    dummy_state = ChatState(messages=[], thread_id="thread1", current_persona="default")

    async def fake_classifier(state, **kwargs):
        # Simulate classifier suggesting secretary
        cl.user_session.set("suggested_persona", {"persona": "secretary", "reason": "User asked about tasks"})
        return {"persona": "secretary", "reason": "User asked about tasks"}

    async def fake_director(state):
        # Simulate director suggesting 'todo' after persona switch
        if state.current_persona == "secretary":
            return ["todo"]
        return ["write"]

    async def fake_todo(state, **kwargs):
        return [AIMessage(content="TODO list updated.", name="todo", metadata={"message_id": "t1"})]

    async def fake_writer(state, **kwargs):
        return [AIMessage(content="Default response.", name="default", metadata={"message_id": "w1"})]

    # Patch underlying functions
    monkeypatch.setattr("src.agents.persona_classifier_agent._classify_persona", fake_classifier)
    monkeypatch.setattr("src.agents.director_agent._direct_actions", fake_director)
    monkeypatch.setattr("src.agents.todo_agent._manage_todo", fake_todo)
    monkeypatch.setattr("src.agents.writer_agent._generate_story", fake_writer)

    # Mock vector store get method
    mock_vector_store = MagicMock()
    mock_vector_store.get = MagicMock(return_value=[])
    mock_vector_store.put = AsyncMock()
    cl.user_session.set("vector_memory", mock_vector_store)

    # Simulate user message triggering the flow
    user_message = HumanMessage(content="Add buy milk", name="Player", metadata={"message_id": "u1"})
    # Manually add user message to state before calling workflow
    dummy_state.messages.append(user_message)
    # Simulate vector store put for user message
    await mock_vector_store.put(content=user_message.content, message_id=user_message.metadata["message_id"], metadata={"type": "human", "author": "Player", "persona": dummy_state.current_persona})

    # Simulate persona switch confirmation (assuming user says 'yes')
    cl.user_session.set("current_persona", "secretary")
    dummy_state.current_persona = "secretary"

    # Call the workflow with the updated state
    result_state = await _chat_workflow(dummy_state.messages, dummy_state)

    # Assertions
    names = [m.name for m in result_state.messages]
    assert "todo" in names
    assert "writer" not in names
    assert result_state.current_persona == "secretary"

    # Check if vector store 'put' was called for the AI message
    mock_vector_store.put.assert_called()
    last_call_args, last_call_kwargs = mock_vector_store.put.call_args_list[-1]
    assert last_call_kwargs['message_id'] == 't1'
    assert last_call_kwargs['metadata']['type'] == 'ai'
    assert last_call_kwargs['metadata']['author'] == 'todo'
    assert last_call_kwargs['metadata']['persona'] == 'secretary'
