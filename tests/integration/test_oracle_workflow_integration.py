import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import uuid # Import uuid for unique thread IDs
from src.models import ChatState, HumanMessage, AIMessage
from src.oracle_workflow import oracle_workflow_runnable  # Use the runnable
from src.config import START_MESSAGE
import chainlit as cl # Import chainlit for mocking

# --- Fixtures ---

@pytest.fixture
def initial_chat_state():
    """Provides a basic initial ChatState with a unique thread_id."""
    thread_id = f"test_thread_{uuid.uuid4()}"
    state = ChatState(
        messages=[AIMessage(content=START_MESSAGE, name="Game Master", metadata={"message_id": f"start_{thread_id}"})], # Add metadata to start message
        thread_id=thread_id,
        current_persona="Storyteller GM" # Start with a known persona
    )
    return state

@pytest.fixture(autouse=True)
def mock_cl_environment_for_oracle(monkeypatch, initial_chat_state):
    """Mocks Chainlit session and message functions for oracle tests."""
    mock_session = MagicMock()
    # Use a copy of the initial state for the session data to avoid test interference
    session_state = initial_chat_state.model_copy(deep=True)
    session_data = {
        "state": session_state,
        "vector_memory": MagicMock(),
        "chat_settings": {"persona": session_state.current_persona, "auto_persona_switch": True},
        "current_persona": session_state.current_persona,
        "user": {"identifier": "test_user"},
        "gm_message": None,
        "suggested_persona": None,
        "pending_persona_switch": None,
        "suppressed_personas": {},
    }

    def mock_get(key, default=None):
        # Ensure the 'state' returned is the one currently in session_data
        if key == 'state':
            return session_data.get('state', default)
        return session_data.get(key, default)

    def mock_set(key, value):
        session_data[key] = value
        # If state is updated, ensure the session_state variable is also updated
        if key == 'state':
            nonlocal session_state
            session_state = value


    mock_session.get = mock_get
    mock_session.set = mock_set
    monkeypatch.setattr(cl, "user_session", mock_session)

    # Mock cl.Message methods to avoid actual UI updates
    mock_message_instance = AsyncMock()
    mock_message_instance.id = f"mock_cl_msg_{uuid.uuid4()}" # Ensure it has a unique ID per instance
    # Make stream_token and send awaitable mocks
    mock_message_instance.stream_token = AsyncMock()
    mock_message_instance.send = AsyncMock()
    mock_message_cls = MagicMock(return_value=mock_message_instance)
    monkeypatch.setattr(cl, "Message", mock_message_cls)

    # Mock cl.context for thread_id
    mock_context = MagicMock()
    # Ensure context thread_id matches the state's thread_id
    mock_context.session.thread_id = initial_chat_state.thread_id
    monkeypatch.setattr(cl, "context", mock_context)

    # Mock storage functions
    monkeypatch.setattr("src.storage.append_log", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.storage.save_text_file", lambda *args, **kwargs: None)
    monkeypatch.setattr("src.storage.load_text_file", lambda *args, **kwargs: "") # Mock load as well

    # Mock vector store put/add methods, allow get to return empty list
    mock_vector_store = session_data["vector_memory"]
    mock_vector_store.put = AsyncMock()
    mock_vector_store.add_documents = AsyncMock()
    mock_vector_store.get = MagicMock(return_value=[])
    # Mock collection add/delete if needed by specific tests (e.g., reset)
    mock_vector_store.collection = AsyncMock()
    mock_vector_store.collection.add = AsyncMock()
    mock_vector_store.collection.delete = AsyncMock()


    # Ensure writer agent uses the mocked cl.Message
    # This might require patching within the writer_agent module specifically

    # Ensure todo agent uses the mocked cl.Message and storage
    monkeypatch.setattr("src.agents.todo_agent.cl", cl)
    monkeypatch.setattr("src.agents.todo_agent.os.path.exists", lambda *args: False) # Assume file doesn't exist initially
    monkeypatch.setattr("src.agents.todo_agent.os.makedirs", lambda *args, **kwargs: None)
    # Mock open carefully if needed, or rely on path.exists mock
    mock_open = MagicMock()
    monkeypatch.setattr("src.agents.todo_agent.open", mock_open)


    return mock_session # Return the mock session

# --- Test Cases ---

@pytest.mark.asyncio
async def test_oracle_calls_writer_once_on_simple_input(initial_chat_state, mock_cl_environment_for_oracle):
    """
    Tests that a simple input triggers the oracle workflow and results in
    a single AI (writer) response being added to the state.
    NOTE: This test makes REAL LLM calls.
    """
    user_input = "I walk into the tavern."
    # Use a copy of the state to avoid modifying the fixture directly
    current_state = initial_chat_state.model_copy(deep=True)
    initial_message_count = len(current_state.messages)

    # Add user message to state
    user_msg = HumanMessage(content=user_input, name="Player", metadata={"message_id": f"user_{uuid.uuid4()}"})
    current_state.messages.append(user_msg)

    # Prepare inputs for the oracle workflow runnable
    # The 'previous' state should reflect the state *before* the user message for context,
    # but the 'messages' list should include the new user message.
    # However, the current implementation of oracle_workflow expects 'previous' to be the full current state.
    inputs = {"messages": current_state.messages, "previous": current_state}

    # Invoke the oracle workflow
    final_state = await oracle_workflow_runnable.ainvoke(inputs, config={"configurable": {"thread_id": current_state.thread_id}})

    # Assertions
    assert isinstance(final_state, ChatState)
    # Find messages added *after* the user message we added
    new_messages = final_state.messages[initial_message_count + 1:] # +1 to skip the user message itself

    # Filter for AI messages that are likely the final writer output
    ai_writer_messages = [
        msg for msg in new_messages
        if isinstance(msg, AIMessage) and msg.name != "error" and not hasattr(msg, "tool_calls") and msg.name != "persona_classifier" # Exclude classifier if it becomes an AIMessage
    ]

    # Check for exactly one writer message
    assert len(ai_writer_messages) == 1, f"Expected exactly one AI writer message, found {len(ai_writer_messages)}: {[m.name for m in ai_writer_messages]}"

    # Check content and persona
    assert "tavern" in ai_writer_messages[0].content.lower(), "AI response should relate to the input"
    assert final_state.current_persona == "Storyteller GM", "Persona should remain Storyteller GM"

    # Verify state was updated in mock session
    final_session_state = mock_cl_environment_for_oracle.get("state")
    # Compare relevant parts, like message count and last message content/name
    assert len(final_session_state.messages) == len(final_state.messages)
    if final_state.messages:
        assert final_session_state.messages[-1].content == final_state.messages[-1].content
        assert final_session_state.messages[-1].name == final_state.messages[-1].name


@pytest.mark.asyncio
async def test_oracle_calls_todo_agent_then_writer(initial_chat_state, mock_cl_environment_for_oracle):
    """
    Tests that input requesting a TODO invokes the todo agent, followed by the writer.
    NOTE: This test makes REAL LLM calls.
    """
    # Use a copy and switch persona to Secretary
    current_state = initial_chat_state.model_copy(deep=True)
    current_state.current_persona = "Secretary"
    mock_cl_environment_for_oracle.set("current_persona", "Secretary")
    mock_cl_environment_for_oracle.set("state", current_state) # Update session state

    user_input = "Please add 'Buy groceries' to my todo list."
    initial_message_count = len(current_state.messages)

    # Add user message to state
    user_msg = HumanMessage(content=user_input, name="Player", metadata={"message_id": f"user_{uuid.uuid4()}"})
    current_state.messages.append(user_msg)

    # Prepare inputs
    inputs = {"messages": current_state.messages, "previous": current_state}

    # Invoke the oracle workflow
    final_state = await oracle_workflow_runnable.ainvoke(inputs, config={"configurable": {"thread_id": current_state.thread_id}})

    # Assertions
    assert isinstance(final_state, ChatState)
    # Get messages added after the user message
    new_messages = final_state.messages[initial_message_count + 1:]

    # Find AI messages added by agents
    ai_messages = [msg for msg in new_messages if isinstance(msg, AIMessage)]

    # Check for a message from the 'todo' agent/tool
    # Note: The todo agent might return content like "Updated TODO list:\n..."
    todo_messages = [msg for msg in ai_messages if msg.name == "todo" or "updated todo list" in msg.content.lower()]
    assert len(todo_messages) >= 1, f"Expected at least one message related to todo, found {[m.name for m in ai_messages]}"
    # Check content more loosely as LLM output varies
    assert "buy groceries" in todo_messages[0].content.lower()

    # Check for exactly one final writer message (Secretary persona)
    # The writer message name includes an icon now, e.g., "🗒️ Secretary"
    writer_messages = [
        msg for msg in ai_messages
        if msg.name is not None and "Secretary" in msg.name and msg.name != "todo" and "updated todo list" not in msg.content.lower() and msg.name != "error"
    ]
    assert len(writer_messages) == 1, f"Expected exactly one final writer message from Secretary, found {len(writer_messages)}: {[m.name for m in ai_messages]}"

    # Ensure the todo message appears *before* the final writer message in the sequence
    try:
        # Find the first todo-related message and the first writer message
        first_todo_msg = next(m for m in new_messages if m.name == "todo" or "updated todo list" in m.content.lower())
        first_writer_msg = next(m for m in new_messages if m.name is not None and "Secretary" in m.name and m.name != "todo" and "updated todo list" not in m.content.lower())

        todo_index = new_messages.index(first_todo_msg)
        writer_index = new_messages.index(first_writer_msg)
        assert todo_index < writer_index, f"Todo message (index {todo_index}) should appear before the final writer message (index {writer_index})"
    except (StopIteration, ValueError):
        pytest.fail(f"Could not find expected todo or writer messages in the final state's new messages: {new_messages}")

    assert final_state.current_persona == "Secretary", "Persona should remain Secretary"


@pytest.mark.asyncio
async def test_persona_switch_confirmation_and_invocation(initial_chat_state, mock_cl_environment_for_oracle):
    """
    Tests the flow: classifier suggests switch, user confirms (mocked),
    new persona workflow runs on the *next* turn.
    NOTE: Uses mocked classifier output, but REAL LLM calls for director/agents.
    """
    # --- Turn 1: Classifier Suggests Switch ---
    current_state_t1 = initial_chat_state.model_copy(deep=True)

    # Mock the classifier to suggest 'Secretary'
    mock_classifier_output = {"persona": "Secretary", "reason": "User asked about lists"}
    # Patch the specific function used within the oracle workflow
    with patch("src.oracle_workflow.persona_classifier_agent", AsyncMock(return_value=mock_classifier_output)) as mock_classifier:

        user_input_turn1 = "Can you help me organize my tasks?"
        initial_message_count_t1 = len(current_state_t1.messages)

        user_msg_t1 = HumanMessage(content=user_input_turn1, name="Player", metadata={"message_id": f"user_t1_{uuid.uuid4()}"})
        current_state_t1.messages.append(user_msg_t1)

        inputs_t1 = {"messages": current_state_t1.messages, "previous": current_state_t1}
        # Invoke with force_classify=True to ensure mocked classifier runs
        inputs_t1["force_classify"] = True
        state_after_turn1 = await oracle_workflow_runnable.ainvoke(inputs_t1, config={"configurable": {"thread_id": current_state_t1.thread_id}})

        # Verify classifier was called
        mock_classifier.assert_called_once()

        # Verify the session now has a pending switch (mocked cl.Message handles the prompt)
        # We check if cl.Message was called with the prompt content
        cl.Message.assert_called() # Check if constructor was called
        call_args_list = cl.Message.call_args_list
        prompt_found = any("suggests switching persona to **Secretary**" in call[1].get('content', '') for call in call_args_list)
        assert prompt_found, "Expected persona switch prompt message"

        # Verify the persona *hasn't* switched yet in the state object itself after turn 1
        assert state_after_turn1.current_persona == "Storyteller GM", f"Persona should still be Storyteller GM after turn 1, but got {state_after_turn1.current_persona}"

    # --- Simulate User Confirmation (outside workflow) ---
    # User says yes, so update the session's current_persona and clear pending
    mock_cl_environment_for_oracle.set("current_persona", "Secretary")
    mock_cl_environment_for_oracle.set("pending_persona_switch", None)
    # Update the state object directly to reflect the confirmed switch for the *next* turn
    state_after_turn1.current_persona = "Secretary"
    mock_cl_environment_for_oracle.set("state", state_after_turn1) # IMPORTANT: Update session state

    # --- Turn 2: New Persona Workflow Runs ---
    user_input_turn2 = "Okay, add 'Write report' to the list."
    initial_message_count_t2 = len(state_after_turn1.messages)

    user_msg_t2 = HumanMessage(content=user_input_turn2, name="Player", metadata={"message_id": f"user_t2_{uuid.uuid4()}"})
    state_after_turn1.messages.append(user_msg_t2) # Add to the state from previous turn

    inputs_t2 = {"messages": state_after_turn1.messages, "previous": state_after_turn1}
    # No need to force_classify here, persona is already set
    final_state = await oracle_workflow_runnable.ainvoke(inputs_t2, config={"configurable": {"thread_id": state_after_turn1.thread_id}})

    # Assertions for Turn 2
    assert final_state.current_persona == "Secretary", "Persona should now be Secretary for turn 2"

    new_messages_t2 = final_state.messages[initial_message_count_t2 + 1:] # Skip user msg
    ai_messages_t2 = [msg for msg in new_messages_t2 if isinstance(msg, AIMessage)]

    # Check that the Secretary's workflow ran (e.g., todo agent + secretary writer)
    todo_messages_t2 = [msg for msg in ai_messages_t2 if msg.name == "todo" or "updated todo list" in msg.content.lower()]
    assert len(todo_messages_t2) >= 1, f"Expected todo agent to run under Secretary persona in turn 2, found {[m.name for m in ai_messages_t2]}"
    assert "write report" in todo_messages_t2[0].content.lower()

    secretary_writer_messages = [
        msg for msg in ai_messages_t2
        if msg.name is not None and "Secretary" in msg.name and msg.name != "todo" and "updated todo list" not in msg.content.lower()
    ]
    assert len(secretary_writer_messages) == 1, f"Expected exactly one Secretary writer message in turn 2, found {len(secretary_writer_messages)}"

# --- Note ---
# Remember that these integration tests make REAL calls to the configured LLM endpoint.
# They will take longer to run than unit tests and may incur API costs.
# Due to the non-deterministic nature of LLMs, occasional failures might occur
# even if the core logic is correct. Rerunning the test or adjusting assertions
# for flexibility (e.g., checking for keywords instead of exact strings) might be needed.
