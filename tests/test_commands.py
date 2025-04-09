import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState
from src.commands import (
    command_roll,
    command_search,
    command_todo,
    command_write,
    command_storyboard,
)
import chainlit as cl # Keep this import

async def mock_send(*args, **kwargs):
    """Mock async send method that returns None."""
    return None

# Fixture for mock state and vector store
@pytest.fixture
def mock_session_data():
    state = ChatState(messages=[], thread_id="cmd-test-thread")
    vector_store = AsyncMock()
    user_session_get = MagicMock(side_effect=lambda key, default=None: {
        "state": state,
        "vector_memory": vector_store,
        "user": {"identifier": "Player"} # Simulate user for commands
    }.get(key, default))
    return state, vector_store, user_session_get

@pytest.mark.asyncio
async def test_command_roll(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    query = "2d6"
    ai_response_msg = AIMessage(content="Rolled 2d6: 7", name="dice_roll", metadata={"message_id": "ai-roll-msg-id"})

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls, \
         patch("src.commands.dice_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_dice_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_instance.id = "user-roll-msg-id"
        mock_cl_message_instance.content = f"/roll {query}"
        mock_cl_message_instance.author = "Player"
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_roll(query)

        mock_cl_message_instance = mock_cl_message_cls.return_value
        assert mock_cl_message_instance.content == f"/roll {query}"
        assert mock_cl_message_instance.author == "Player"
        mock_cl_message_instance.send.assert_awaited_once_with()
        mock_dice_agent.assert_awaited_once_with(state)

        # Check state update
        assert len(state.messages) == 2
        assert isinstance(state.messages[0], HumanMessage)
        assert state.messages[0].content == f"/roll {query}"
        assert state.messages[0].metadata["message_id"] == "user-roll-msg-id"
        assert state.messages[1] == ai_response_msg

        # Check vector store puts
        assert vector_store.put.await_count == 2
        vector_store.put.assert_any_await(content=f"/roll {query}", message_id="user-roll-msg-id", metadata={'type': 'human', 'author': 'Player', 'persona': 'Default'})
        vector_store.put.assert_any_await(content="Rolled 2d6: 7", message_id="ai-roll-msg-id", metadata={'type': 'ai', 'author': 'dice_roll'})

        mock_user_session_set.assert_called_with("state", state)

@pytest.mark.asyncio
async def test_command_search(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    query = "history of dragons"
    ai_response_msg = AIMessage(content="Search results...", name="web_search", metadata={"message_id": "ai-search-msg-id"})

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls, \
         patch("src.commands.web_search_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_search_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_instance.id = "user-search-msg-id"
        mock_cl_message_instance.content = f"/search {query}"
        mock_cl_message_instance.author = "Player"
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_search(query)

        mock_cl_message_instance = mock_cl_message_cls.return_value
        assert mock_cl_message_instance.content == f"/search {query}"
        assert mock_cl_message_instance.author == "Player"
        mock_cl_message_instance.send.assert_awaited_once_with()
        mock_search_agent.assert_awaited_once_with(state)

        assert len(state.messages) == 2
        assert state.messages[1] == ai_response_msg
        assert vector_store.put.await_count == 2
        vector_store.put.assert_any_await(content=f"/search {query}", message_id="user-search-msg-id", metadata={'type': 'human', 'author': 'Player', 'persona': 'Default'})
        vector_store.put.assert_any_await(content="Search results...", message_id="ai-search-msg-id", metadata={'type': 'ai', 'author': 'web_search'})

        mock_user_session_set.assert_called_with("state", state)


@pytest.mark.asyncio
async def test_command_todo(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    query = "buy milk"
    ai_response_msg = AIMessage(content="Added: buy milk", name="todo", metadata={"message_id": "ai-todo-msg-id"})

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls, \
         patch("src.commands.call_todo_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_call_todo_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_instance.id = "user-todo-msg-id"
        mock_cl_message_instance.content = f"/todo {query}"
        mock_cl_message_instance.author = "Player"
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_todo(query)

        mock_cl_message_instance = mock_cl_message_cls.return_value
        assert mock_cl_message_instance.content == f"/todo {query}"
        assert mock_cl_message_instance.author == "Player"
        # The /todo command triggers TWO sends: one for user message, one for AI reply
        assert mock_cl_message_instance.send.await_count == 2
        mock_call_todo_agent.assert_awaited_once_with(state)

        assert len(state.messages) == 2
        assert state.messages[1] == ai_response_msg
        assert vector_store.put.await_count == 2
        vector_store.put.assert_any_await(content=f"/todo {query}", message_id="user-todo-msg-id", metadata={'type': 'human', 'author': 'Player', 'persona': 'Default'})
        vector_store.put.assert_any_await(content="Added: buy milk", message_id="ai-todo-msg-id", metadata={'type': 'ai', 'author': 'todo'})

        mock_user_session_set.assert_called_with("state", state)

@pytest.mark.asyncio
async def test_command_write(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    query = "the wizard speaks"
    ai_response_msg = AIMessage(content="The wizard says hello.", name="Game Master", metadata={"message_id": "ai-write-msg-id"})

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls, \
         patch("src.agents.writer_agent.call_writer_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_call_writer_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_instance.id = "user-write-msg-id"
        mock_cl_message_instance.content = f"/write {query}"
        mock_cl_message_instance.author = "Player"
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_write(query)

        mock_cl_message_instance = mock_cl_message_cls.return_value
        assert mock_cl_message_instance.content == f"/write {query}"
        assert mock_cl_message_instance.author == "Player"
        mock_cl_message_instance.send.assert_awaited_once_with()
        mock_call_writer_agent.assert_awaited_once_with(state)

        assert len(state.messages) == 2
        assert state.messages[1] == ai_response_msg
        assert vector_store.put.await_count == 2
        vector_store.put.assert_any_await(content=f"/write {query}", message_id="user-write-msg-id", metadata={'type': 'human', 'author': 'Player', 'persona': 'Default'})
        vector_store.put.assert_any_await(content="The wizard says hello.", message_id="ai-write-msg-id", metadata={'type': 'ai', 'author': 'Game Master'})

        mock_user_session_set.assert_called_with("state", state)


@pytest.mark.asyncio
async def test_command_storyboard_enabled(mock_session_data):
    state, _, user_session_get = mock_session_data
    # Add a GM message to the state
    gm_msg = AIMessage(content="Last scene description", name="Game Master", metadata={"message_id": "gm-msg-for-storyboard"})
    state.messages.append(gm_msg)

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls, \
         patch("src.commands.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_agent, \
         patch("src.commands.IMAGE_GENERATION_ENABLED", True): # Ensure enabled

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_storyboard()

        # Check that the "Generating..." message was sent
        mock_cl_message_cls.assert_called_with(content="Generating storyboard for the last scene...")
        mock_cl_message_instance.send.assert_awaited_once_with()

        # Check that the agent was called with the correct state and message ID
        mock_storyboard_agent.assert_awaited_once_with(state=state, gm_message_id="gm-msg-for-storyboard")

@pytest.mark.asyncio
async def test_command_storyboard_disabled(mock_session_data):
    state, _, user_session_get = mock_session_data

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls, \
         patch("src.commands.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_agent, \
         patch("src.commands.IMAGE_GENERATION_ENABLED", False): # Ensure disabled

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_storyboard()

        # Check that the "disabled" message was sent
        mock_cl_message_cls.assert_called_with(content="Image generation is disabled.")
        mock_cl_message_instance.send.assert_awaited_once_with()

        # Check that the agent was NOT called
        mock_storyboard_agent.assert_not_awaited()

@pytest.mark.asyncio
async def test_command_storyboard_no_gm_message(mock_session_data):
    state, _, user_session_get = mock_session_data
    # State has no GM messages or GM messages without metadata
    state.messages.append(HumanMessage(content="Just a human message"))
    state.messages.append(AIMessage(content="AI message, but not GM", name="other_ai"))
    state.messages.append(AIMessage(content="GM message, no metadata", name="Game Master", metadata={}))


    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls, \
         patch("src.commands.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_agent, \
         patch("src.commands.IMAGE_GENERATION_ENABLED", True): # Ensure enabled

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_storyboard()

        # Check that the "Could not find" message was sent
        mock_cl_message_cls.assert_called_with(content="Could not find a previous Game Master message with a valid ID to generate a storyboard for.")
        mock_cl_message_instance.send.assert_awaited_once_with()

        # Check that the agent was NOT called
        mock_storyboard_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_command_missing_state():
    # Simulate user_session.get returning None for state
    user_session_get = MagicMock(return_value=None)
    query = "anything"

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("chainlit.Message", new_callable=MagicMock) as mock_cl_message_cls:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        # Test one command, the logic is the same for all
        await command_roll(query)

        # Check that the "Error: Session state not found." message was sent
        mock_cl_message_cls.assert_called_with(content="Error: Session state not found.")
        mock_cl_message_instance.send.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_command_help():
    with patch("src.commands.cl.Message", new_callable=MagicMock) as mock_cl_message_cls:
        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        from src.commands import command_help
        await command_help()

        mock_cl_message_instance.send.assert_awaited_once()
        sent_content = mock_cl_message_cls.call_args.kwargs.get("content", "")
        assert "Available Commands" in sent_content

@pytest.mark.asyncio
async def test_command_reset(mock_session_data):
    state, vector_store, user_session_get = mock_session_data

    dummy_context = MagicMock()
    dummy_context.session.thread_id = "cmd-test-thread"

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("src.commands.cl.context", dummy_context), \
         patch("src.commands.cl.Message", new_callable=MagicMock) as mock_cl_message_cls:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_instance.id = "start-msg-id"
        mock_cl_message_cls.return_value = mock_cl_message_instance

        # Patch vector_store.collection.delete to track calls
        vector_store.collection = AsyncMock()
        vector_store.collection.delete = AsyncMock()

        from src.commands import command_reset
        await command_reset()

        mock_cl_message_instance.send.assert_awaited_once()
        args, kwargs = mock_user_session_set.call_args_list[-1]
        assert args[0] == "state"
        new_state = args[1]
        assert new_state.messages
        assert isinstance(new_state.messages[0], AIMessage)
        assert new_state.messages[0].content.startswith("Hello")

        # Vector store cleared
        vector_store.collection.delete.assert_awaited_once()

@pytest.mark.asyncio
async def test_command_save(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    # Add some messages
    state.messages.append(HumanMessage(content="Hi", name="Player"))
    state.messages.append(AIMessage(content="Hello", name="Game Master"))

    # Patch chainlit.element.context BEFORE importing command_save
    from chainlit import element as cl_element
    dummy_context = MagicMock()
    dummy_context.session.thread_id = "cmd-test-thread"
    cl_element.context = dummy_context

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.context", dummy_context), \
         patch("src.commands.cl.Message", new_callable=MagicMock) as mock_cl_message_cls:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        from src.commands import command_save
        await command_save()

        mock_cl_message_instance.send.assert_awaited_once()
        sent_elements = mock_cl_message_cls.call_args.kwargs.get("elements", [])
        assert sent_elements
        file_element = sent_elements[0]
        assert file_element.name.endswith(".md")
        assert b"Player" in file_element.content


@pytest.mark.asyncio
async def test_unknown_slash_command():
    with patch("src.commands.cl.Message", new_callable=MagicMock) as mock_cl_message_cls:
        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        from src.event_handlers import on_message
        dummy_msg = MagicMock()
        dummy_msg.content = "/unknowncmd foo"
        dummy_msg.command = ""  # Not a button
        dummy_msg.author = "Player"
        dummy_msg.id = "msgid"

        # Patch user session to have state and vector store
        with patch("src.event_handlers.cl.user_session.get", side_effect=lambda k, default=None: {
            "state": ChatState(messages=[], thread_id="t1"),
            "vector_memory": AsyncMock(),
            "user": {"identifier": "Player"}
        }.get(k, default)):
            await on_message(dummy_msg)

        mock_cl_message_cls.assert_called_with(content="Unknown command: /unknowncmd")
        mock_cl_message_instance.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_empty_slash_command():
    with patch("src.commands.cl.Message", new_callable=MagicMock) as mock_cl_message_cls:
        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        from src.event_handlers import on_message
        dummy_msg = MagicMock()
        dummy_msg.content = "/"
        dummy_msg.command = ""  # Not a button
        dummy_msg.author = "Player"
        dummy_msg.id = "msgid"

        with patch("src.event_handlers.cl.user_session.get", side_effect=lambda k, default=None: {
            "state": ChatState(messages=[], thread_id="t1"),
            "vector_memory": AsyncMock(),
            "user": {"identifier": "Player"}
        }.get(k, default)):
            await on_message(dummy_msg)

        mock_cl_message_cls.assert_called_with(content="Unknown command: /")
        mock_cl_message_instance.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_command_save_empty_state():
    from src.commands import command_save
    with patch("src.commands.cl.user_session.get", return_value=None), \
         patch("src.commands.cl.Message", new_callable=MagicMock) as mock_cl_message_cls:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.send.return_value = None
        mock_cl_message_cls.return_value = mock_cl_message_instance

        await command_save()

        mock_cl_message_cls.assert_called_with(content="No story to save.")
        mock_cl_message_instance.send.assert_awaited_once()
