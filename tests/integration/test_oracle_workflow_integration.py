import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
import uuid
from src.models import ChatState, HumanMessage, AIMessage, ToolMessage
from src.oracle_workflow import oracle_workflow
from src.config import START_MESSAGE
import chainlit as cl
import src.oracle_workflow  # <-- Add this import for src.oracle_workflow

@pytest.fixture
def initial_chat_state():
    """Provides a basic initial ChatState with a unique thread_id."""
    thread_id = f"test_thread_{uuid.uuid4()}"
    state = ChatState(
        messages=[AIMessage(content=START_MESSAGE, name="Game Master", metadata={"message_id": f"start_{thread_id}", "persona": "storyteller_gm"})],
        thread_id=thread_id,
        current_persona="storyteller_gm"
    )
    return state

@pytest.fixture(autouse=True)
def mock_cl_environment_for_oracle(monkeypatch, initial_chat_state):
    import src.agents.persona_classifier_agent as pca_module  # Use alias
    import src.agents.director_agent as director_module
    import src.agents.knowledge_agent as knowledge_module
    import src.agents.writer_agent as writer_module
    import src.agents.todo_agent as todo_module
    import src.agents.dice_agent as dice_module
    import src.event_handlers as event_handlers_module
    import src.agents.web_search_agent as web_search_module
    import src.agents.report_agent as report_module
    import src.agents.storyboard_editor_agent as storyboard_module
    import src.oracle_workflow as oracle_workflow_module

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
        if key == 'state':
            return session_data.get('state', default)
        if key == "gm_message":
            mock_msg = MagicMock(spec=cl.Message)
            mock_msg.stream_token = AsyncMock()
            mock_msg.send = AsyncMock()
            mock_msg.update = AsyncMock()
            mock_msg.id = f"mock_gm_msg_{uuid.uuid4()}"
            return mock_msg
        return session_data.get(key, default)

    def mock_set(key, value):
        session_data[key] = value
        if key == 'state':
            nonlocal session_state
            session_state = value

    mock_user_session = MagicMock()
    mock_user_session.get = mock_get
    mock_user_session.set = mock_set
    monkeypatch.setattr(cl, "user_session", mock_user_session)
    # Patch cl object within each module that uses `import chainlit as cl`
    # Patch cl.user_session where it's directly imported as 'cl'
    monkeypatch.setattr(oracle_workflow_module.cl, "user_session", mock_user_session, raising=False)
    monkeypatch.setattr(pca_module.cl, "user_session", mock_user_session, raising=False)
    # monkeypatch.setattr(director_module.cl, "user_session", mock_user_session, raising=False) # director_agent removed
    monkeypatch.setattr(knowledge_module.cl, "user_session", mock_user_session, raising=False)
    monkeypatch.setattr(writer_module.cl, "user_session", mock_user_session, raising=False)
    monkeypatch.setattr(todo_module.cl, "user_session", mock_user_session, raising=False)
    monkeypatch.setattr(report_module.cl, "user_session", mock_user_session, raising=False)
    monkeypatch.setattr(storyboard_module.cl, "user_session", mock_user_session, raising=False)
    monkeypatch.setattr(web_search_module.cl, "user_session", mock_user_session, raising=False) # Added for web_search_agent

    # Patch cl.user_session where it's imported as 'cl_user_session'
    monkeypatch.setattr(dice_module, "cl_user_session", mock_user_session, raising=False)
    monkeypatch.setattr(event_handlers_module, "cl_user_session", mock_user_session, raising=False)

    # Patch cl.Message where it's directly imported as 'cl'
    mock_message_instance = AsyncMock(spec=cl.Message)
    mock_message_instance.id = f"mock_cl_msg_{uuid.uuid4()}"
    mock_message_instance.stream_token = AsyncMock()
    mock_message_instance.send = AsyncMock()
    mock_message_instance.update = AsyncMock()
    mock_message_cls = MagicMock(return_value=mock_message_instance)
    monkeypatch.setattr(cl, "Message", mock_message_cls) # General patch
    # Patch cl.Message where it's imported as 'cl'
    monkeypatch.setattr(oracle_workflow_module.cl, "Message", mock_message_cls, raising=False)
    monkeypatch.setattr(pca_module.cl, "Message", mock_message_cls, raising=False)
    # monkeypatch.setattr(director_module.cl, "Message", mock_message_cls, raising=False) # director_agent removed
    monkeypatch.setattr(knowledge_module.cl, "Message", mock_message_cls, raising=False)
    monkeypatch.setattr(writer_module.cl, "Message", mock_message_cls, raising=False)
    monkeypatch.setattr(todo_module.cl, "Message", mock_message_cls, raising=False)
    monkeypatch.setattr(report_module.cl, "Message", mock_message_cls, raising=False)
    monkeypatch.setattr(storyboard_module.cl, "Message", mock_message_cls, raising=False)
    monkeypatch.setattr(web_search_module.cl, "Message", mock_message_cls, raising=False) # Added for web_search_agent

    # Patch cl.Message where it's imported as 'CLMessage'
    monkeypatch.setattr(todo_module, "CLMessage", mock_message_cls, raising=False)
    monkeypatch.setattr(dice_module, "CLMessage", mock_message_cls, raising=False)
    monkeypatch.setattr(web_search_module, "CLMessage", mock_message_cls, raising=False) # Use alias
    monkeypatch.setattr(report_module, "CLMessage", mock_message_cls, raising=False)
    monkeypatch.setattr(storyboard_module, "CLMessage", mock_message_cls, raising=False)
    monkeypatch.setattr(writer_module, "cl", MagicMock(Message=mock_message_cls, user_session=mock_user_session, AsyncLangchainCallbackHandler=AsyncMock()), raising=False) # Keep writer patch for cb

    # Patch cl.AsyncLangchainCallbackHandler in writer_agent
    monkeypatch.setattr(writer_module.cl, "AsyncLangchainCallbackHandler", AsyncMock())


    monkeypatch.setattr(oracle_workflow_module, "append_log", lambda *args, **kwargs: None)
    monkeypatch.setattr(oracle_workflow_module, "save_text_file", lambda *args, **kwargs: None)
    monkeypatch.setattr(oracle_workflow_module, "get_persona_daily_dir", lambda *a, **kw: MagicMock(joinpath=lambda x: MagicMock()))

    mock_vector_store = session_data["vector_memory"]
    mock_vector_store.put = AsyncMock()
    mock_vector_store.add_documents = AsyncMock()
    mock_vector_store.get = MagicMock(return_value=[])
    mock_vector_store.collection = AsyncMock()
    mock_vector_store.collection.add = AsyncMock()
    mock_vector_store.collection.delete = AsyncMock()

    monkeypatch.setattr("src.agents.todo_agent.os.path.exists", lambda *args: False)
    monkeypatch.setattr("src.agents.todo_agent.os.makedirs", lambda *args, **kwargs: None)
    mock_open = MagicMock()
    monkeypatch.setattr("builtins.open", mock_open)

    monkeypatch.setattr(oracle_workflow_module, "persona_classifier_agent", AsyncMock(return_value={"persona": initial_chat_state.current_persona}))

    return mock_user_session

@pytest.mark.asyncio
async def test_oracle_single_step_persona_agent(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
    user_input = "Tell me a story."
    current_state = initial_chat_state.model_copy(deep=True)
    user_msg = HumanMessage(content=user_input, name="Player", metadata={"message_id": f"user_{uuid.uuid4()}"})
    current_state.messages.append(user_msg)
    initial_message_count = len(current_state.messages)

    # Patch the oracle_agent function within the oracle_workflow module
    mock_oracle_agent = AsyncMock(return_value=current_state.current_persona)
    monkeypatch.setattr(
        "src.oracle_workflow.oracle_agent",
        mock_oracle_agent
    )

    mock_persona_agent_output = [AIMessage(content="A grand story unfolds!", name=current_state.current_persona, metadata={"message_id": "agent_msg_1", "agent": current_state.current_persona, "persona": current_state.current_persona})]
    mock_persona_agent = AsyncMock(return_value=mock_persona_agent_output)
    monkeypatch.setitem(
        src.oracle_workflow.agents_map,
        current_state.current_persona,
        mock_persona_agent
    )

    inputs = {"messages": current_state.messages, "previous": current_state, "state": current_state}

    final_state = await oracle_workflow(inputs, current_state, config={"configurable": {"thread_id": current_state.thread_id}})

    assert isinstance(final_state, ChatState)
    mock_oracle_agent.assert_called_once() # Check if oracle_agent was called
    mock_persona_agent.assert_called_once()
    assert len(final_state.messages) == initial_message_count + 1
    last_message = final_state.messages[-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.content == "A grand story unfolds!"
    assert last_message.name == current_state.current_persona
    # Check metadata added by the oracle loop
    assert last_message.metadata.get("agent") == current_state.current_persona
    assert last_message.metadata.get("persona") == current_state.current_persona

@pytest.mark.asyncio
async def test_oracle_multi_step_tool_then_persona(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
    user_input = "Search for dragons then tell me a story."
    current_state = initial_chat_state.model_copy(deep=True)
    user_msg = HumanMessage(content=user_input, name="Player", metadata={"message_id": f"user_{uuid.uuid4()}"})
    current_state.messages.append(user_msg)
    initial_message_count = len(current_state.messages)

    oracle_decisions = ["search", current_state.current_persona] # Oracle decides sequence
    mock_oracle_agent = AsyncMock(side_effect=oracle_decisions)
    monkeypatch.setattr(
        "src.oracle_workflow.oracle_agent",
        mock_oracle_agent
    )

    mock_search_output = [AIMessage(content="Found info about dragons.", name="search_tool", metadata={"message_id": "search_msg_1", "agent": "search", "persona": current_state.current_persona})]
    mock_search_agent = AsyncMock(return_value=mock_search_output)
    monkeypatch.setitem(src.oracle_workflow.agents_map, "search", mock_search_agent)

    mock_persona_agent_output = [AIMessage(content="Okay, here's a story about dragons...", name=current_state.current_persona, metadata={"message_id": "agent_msg_1", "agent": current_state.current_persona, "persona": current_state.current_persona})]
    mock_persona_agent = AsyncMock(return_value=mock_persona_agent_output)
    monkeypatch.setitem(src.oracle_workflow.agents_map, current_state.current_persona, mock_persona_agent)

    inputs = {"messages": current_state.messages, "previous": current_state, "state": current_state}

    final_state = await oracle_workflow(inputs, current_state, config={"configurable": {"thread_id": current_state.thread_id}})

    assert isinstance(final_state, ChatState)
    assert mock_oracle_agent.call_count == 2 # Called for search, then for persona agent
    mock_search_agent.assert_called_once()
    mock_persona_agent.assert_called_once()

    assert len(final_state.messages) == initial_message_count + 2
    search_message = final_state.messages[-2]
    persona_message = final_state.messages[-1]

    assert isinstance(search_message, AIMessage)
    assert search_message.content == "Found info about dragons."
    # Check metadata added by the oracle loop
    assert search_message.metadata.get("agent") == "search"
    assert search_message.metadata.get("persona") == current_state.current_persona


    assert isinstance(persona_message, AIMessage)
    assert persona_message.content == "Okay, here's a story about dragons..."
    assert persona_message.name == current_state.current_persona
    # Check metadata added by the oracle loop
    assert persona_message.metadata.get("agent") == current_state.current_persona
    assert persona_message.metadata.get("persona") == current_state.current_persona

@pytest.mark.asyncio
async def test_oracle_max_iterations_reached(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
    user_input = "Keep searching."
    current_state = initial_chat_state.model_copy(deep=True)
    user_msg = HumanMessage(content=user_input, name="Player", metadata={"message_id": f"user_{uuid.uuid4()}"})
    current_state.messages.append(user_msg)
    initial_message_count = len(current_state.messages)

    # Use the actual config value for max iterations
    from src.config import MAX_CHAIN_LENGTH as MAX_ITER

    mock_oracle_agent = AsyncMock(return_value="search") # Oracle keeps saying search
    monkeypatch.setattr("src.oracle_workflow.oracle_agent", mock_oracle_agent)

    # Mock search agent to return unique messages
    mock_search_outputs = [
        [AIMessage(content=f"Still searching... {i+1}", name="search_tool", metadata={"message_id": f"search_msg_{i+1}", "agent": "search", "persona": current_state.current_persona})]
        for i in range(MAX_ITER)
    ]
    mock_search_agent = AsyncMock(side_effect=mock_search_outputs)
    monkeypatch.setitem(src.oracle_workflow.agents_map, "search", mock_search_agent)

    inputs = {"messages": current_state.messages, "previous": current_state, "state": current_state}

    final_state = await oracle_workflow(inputs, current_state, config={"configurable": {"thread_id": current_state.thread_id}})

    assert isinstance(final_state, ChatState)
    # Oracle is called MAX_ITER times, then loop breaks
    assert mock_oracle_agent.call_count == MAX_ITER
    assert mock_search_agent.call_count == MAX_ITER
    # Should have initial messages + MAX_ITER search results + 1 max iteration system message
    assert len(final_state.messages) == initial_message_count + MAX_ITER + 1
    # Check for the max iteration message
    assert final_state.messages[-1].name == "system"
    assert "maximum processing steps" in final_state.messages[-1].content

@pytest.mark.asyncio
async def test_oracle_persona_classification_updates_state(initial_chat_state, mock_cl_environment_for_oracle, monkeypatch):
    user_input = "Let's switch gears and talk about my feelings."
    current_state = initial_chat_state.model_copy(deep=True)
    current_state.current_persona = "storyteller_gm"
    user_msg = HumanMessage(content=user_input, name="Player", metadata={"message_id": f"user_{uuid.uuid4()}"})
    current_state.messages.append(user_msg)

    mock_classifier = AsyncMock(return_value={"persona": "therapist", "reason": "User mentioned feelings"})
    monkeypatch.setattr("src.oracle_workflow.persona_classifier_agent", mock_classifier)

    # Oracle should be called *after* classification updates the state
    mock_oracle_agent = AsyncMock(return_value="therapist")
    monkeypatch.setattr("src.oracle_workflow.oracle_agent", mock_oracle_agent)

    mock_therapist_output = [AIMessage(content="Let's talk about that.", name="therapist", metadata={"message_id": "therapist_msg_1", "agent": "therapist", "persona": "therapist"})]
    mock_therapist_agent = AsyncMock(return_value=mock_therapist_output)
    monkeypatch.setitem(src.oracle_workflow.agents_map, "therapist", mock_therapist_agent)

    inputs = {"messages": current_state.messages, "previous": current_state, "state": current_state, "force_classify": True}

    final_state = await oracle_workflow(inputs, current_state, config={"configurable": {"thread_id": current_state.thread_id}})

    assert isinstance(final_state, ChatState)
    mock_classifier.assert_called_once()
    mock_oracle_agent.assert_called_once() # Oracle is called once
    mock_therapist_agent.assert_called_once()

    # Check that the state passed to the Oracle agent had the updated persona
    oracle_call_args, oracle_call_kwargs = mock_oracle_agent.call_args
    state_passed_to_oracle = oracle_call_args[0]
    assert isinstance(state_passed_to_oracle, ChatState)
    assert state_passed_to_oracle.current_persona == "therapist"

    assert final_state.current_persona == "therapist"
    last_message = final_state.messages[-1]
    assert isinstance(last_message, AIMessage)
    assert last_message.name == "therapist"
    # Check metadata added by the oracle loop
    assert last_message.metadata.get("agent") == "therapist"
    assert last_message.metadata.get("persona") == "therapist"
