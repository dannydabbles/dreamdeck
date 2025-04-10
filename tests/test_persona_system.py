import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call  # Added MagicMock, call
from src.agents.persona_classifier_agent import _classify_persona, PERSONA_LIST
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage
import chainlit as cl
import uuid  # Add uuid import

from tests.test_event_handlers import mock_cl_environment  # Ensure this line exists and is correct

from src.event_handlers import on_message
from src.agents.writer_agent import _generate_story, call_writer_agent
from src.agents.writer_agent import _WriterAgentWrapper  # For patching __call__
from src.agents.director_agent import _direct_actions, director_agent
# Import the compiled LangGraph app *without* the checkpointer for these tests
from src.workflows import app_without_checkpoint as chat_workflow_app
from src.agents.dice_agent import _dice_roll, _DiceAgentWrapper  # For patching __call__
from src.agents.todo_agent import _manage_todo, manage_todo  # For patching


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
        # Return a simple dict to test serialization
        return {"messages": [AIMessage(content="Therapy response", name="🤖 therapist", metadata={"message_id": "w1"})]}

    # Patch the underlying functions
    monkeypatch.setattr("src.workflows.director_agent", fake_director)
    # Patch the @task decorated dice_roll function
    monkeypatch.setattr("src.agents.dice_agent.dice_roll", fake_dice)
    # Patch the @task decorated generate_story function
    monkeypatch.setattr("src.agents.writer_agent.generate_story", fake_writer)

    # Mock vector store get method if needed
    mock_vector_store = MagicMock()
    mock_vector_store.get = MagicMock(return_value=[])
    mock_vector_store.put = AsyncMock()
    cl.user_session.set("vector_memory", mock_vector_store)

    # Provide unique thread_id for this test
    import uuid as _uuid
    test_thread_id = f"test_thread_{_uuid.uuid4()}"
    workflow_config = {"configurable": {"thread_id": test_thread_id}}

    # Construct the 'previous' state expected by the entrypoint
    previous_state = ChatState(
        messages=[],
        thread_id=test_thread_id,
        current_persona=dummy_state.current_persona
    )
    input_data = {
        "messages": dummy_state.messages,
        "previous": previous_state
    }
    final_state_obj = await chat_workflow_app.ainvoke(input_data, config=workflow_config)

    if isinstance(final_state_obj, ChatState):
        result_messages = final_state_obj.messages
    else:
        raise TypeError(f"Expected ChatState, got {type(final_state_obj)}")

    # Check if the fake_writer output was merged into the state messages
    therapist_msg_found = False
    for msg in result_messages:
        if isinstance(msg, AIMessage) and msg.name in ("Therapist", "🤖 therapist"):
            therapist_msg_found = True
            break
    assert therapist_msg_found, f"Expected Therapist message in {result_messages}"


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
        # Return a simple dict to test serialization
        return {"messages": [AIMessage(content="TODO list updated.", name="todo", metadata={"message_id": "t1"})]}

    async def fake_writer(state, **kwargs):
        # Return a simple dict to test serialization
        return {"messages": [AIMessage(content="Default response.", name="default", metadata={"message_id": "w1"})]}

    # Patch underlying functions
    monkeypatch.setattr("src.event_handlers.persona_classifier_agent", fake_classifier)
    monkeypatch.setattr("src.workflows.director_agent", fake_director)
    # Patch the specific manage_todo function
    monkeypatch.setattr("src.agents.todo_agent.manage_todo", fake_todo)
    # Patch the @task decorated generate_story function
    monkeypatch.setattr("src.agents.writer_agent.generate_story", fake_writer)

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
    # Do NOT modify dummy_state.current_persona directly; instead, pass persona explicitly in input_data

    # Call the workflow with the updated state
    # Provide unique thread_id for this test
    import uuid as _uuid
    test_thread_id = f"test_thread_{_uuid.uuid4()}"
    workflow_config = {"configurable": {"thread_id": test_thread_id}}

    # Construct the 'previous' state reflecting the persona switch
    previous_state = ChatState(
        messages=[],
        thread_id=test_thread_id,
        current_persona="secretary"
    )
    input_data = {
        "messages": dummy_state.messages,
        "previous": previous_state
    }
    final_state_obj = await chat_workflow_app.ainvoke(input_data, config=workflow_config)

    if isinstance(final_state_obj, ChatState):
        result_messages = final_state_obj.messages
        final_persona = final_state_obj.current_persona
    else:
        raise TypeError(f"Expected ChatState, got {type(final_state_obj)}")

    # Assertions
    # Check if the fake_todo output was merged into the state messages
    names = [m.name for m in result_messages if isinstance(m, AIMessage)]
    assert "todo" in names, f"Expected 'todo' in AI message names: {names}"
    assert "writer" not in names
    assert final_persona == "secretary", f"Expected final persona to be 'secretary', but got '{final_persona}'"

    # Check if vector store 'put' was called for the AI message
    mock_vector_store.put.assert_called()
    last_call_args, last_call_kwargs = mock_vector_store.put.call_args_list[-1]
    try:
        import uuid as _uuid
    except ImportError:
        import uuid as _uuid
    try:
        _uuid.UUID(last_call_kwargs['message_id'])
    except ValueError:
        assert False, f"message_id is not a valid UUID: {last_call_kwargs['message_id']}"
    assert last_call_kwargs['metadata']['type'] == 'ai'
    assert last_call_kwargs['metadata']['author'] == 'todo'
    assert last_call_kwargs['metadata']['persona'] == 'secretary'
