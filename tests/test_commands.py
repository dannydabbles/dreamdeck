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
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls, \
         patch("src.commands.dice_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_dice_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.id = "user-roll-msg-id"
        mock_cl_message_cls.return_value = mock_cl_message_instance
        mock_cl_message_instance.send = AsyncMock()

        await command_roll(query)

        mock_cl_message_cls.assert_called_with(content=f"/roll {query}", author="Player")
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
        vector_store.put.assert_any_await(content=f"/roll {query}", message_id="user-roll-msg-id", metadata={'type': 'human', 'author': 'Player'})
        vector_store.put.assert_any_await(content="Rolled 2d6: 7", message_id="ai-roll-msg-id", metadata={'type': 'ai', 'author': 'dice_roll'})

        mock_user_session_set.assert_called_with("state", state)

@pytest.mark.asyncio
async def test_command_search(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    query = "history of dragons"
    ai_response_msg = AIMessage(content="Search results...", name="web_search", metadata={"message_id": "ai-search-msg-id"})

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls, \
         patch("src.commands.web_search_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_search_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.id = "user-search-msg-id"
        mock_cl_message_cls.return_value = mock_cl_message_instance
        mock_cl_message_instance.send = AsyncMock()

        await command_search(query)

        mock_cl_message_cls.assert_called_with(content=f"/search {query}", author="Player")
        mock_cl_message_instance.send.assert_awaited_once_with()
        mock_search_agent.assert_awaited_once_with(state)

        assert len(state.messages) == 2
        assert state.messages[1] == ai_response_msg
        assert vector_store.put.await_count == 2
        vector_store.put.assert_any_await(content=f"/search {query}", message_id="user-search-msg-id", metadata={'type': 'human', 'author': 'Player'})
        vector_store.put.assert_any_await(content="Search results...", message_id="ai-search-msg-id", metadata={'type': 'ai', 'author': 'web_search'})

        mock_user_session_set.assert_called_with("state", state)


@pytest.mark.asyncio
async def test_command_todo(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    query = "buy milk"
    ai_response_msg = AIMessage(content="Added: buy milk", name="todo", metadata={"message_id": "ai-todo-msg-id"})

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls, \
         patch("src.commands.todo_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_todo_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.id = "user-todo-msg-id"
        mock_cl_message_cls.return_value = mock_cl_message_instance
        mock_cl_message_instance.send = AsyncMock()

        await command_todo(query)

        mock_cl_message_cls.assert_called_with(content=f"/todo {query}", author="Player")
        mock_cl_message_instance.send.assert_awaited_once_with()
        mock_todo_agent.assert_awaited_once_with(state)

        assert len(state.messages) == 2
        assert state.messages[1] == ai_response_msg
        assert vector_store.put.await_count == 2
        vector_store.put.assert_any_await(content=f"/todo {query}", message_id="user-todo-msg-id", metadata={'type': 'human', 'author': 'Player'})
        vector_store.put.assert_any_await(content="Added: buy milk", message_id="ai-todo-msg-id", metadata={'type': 'ai', 'author': 'todo'})

        mock_user_session_set.assert_called_with("state", state)

@pytest.mark.asyncio
async def test_command_write(mock_session_data):
    state, vector_store, user_session_get = mock_session_data
    query = "the wizard speaks"
    ai_response_msg = AIMessage(content="The wizard says hello.", name="Game Master", metadata={"message_id": "ai-write-msg-id"})

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("src.commands.cl.user_session.set", new_callable=MagicMock) as mock_user_session_set, \
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls, \
         patch("src.commands.writer_agent", new_callable=AsyncMock, return_value=[ai_response_msg]) as mock_writer_agent:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_instance.id = "user-write-msg-id"
        mock_cl_message_cls.return_value = mock_cl_message_instance
        mock_cl_message_instance.send = AsyncMock()

        await command_write(query)

        mock_cl_message_cls.assert_called_with(content=f"/write {query}", author="Player")
        mock_cl_message_instance.send.assert_awaited_once_with()
        mock_writer_agent.assert_awaited_once_with(state)

        assert len(state.messages) == 2
        assert state.messages[1] == ai_response_msg
        assert vector_store.put.await_count == 2
        vector_store.put.assert_any_await(content=f"/write {query}", message_id="user-write-msg-id", metadata={'type': 'human', 'author': 'Player'})
        vector_store.put.assert_any_await(content="The wizard says hello.", message_id="ai-write-msg-id", metadata={'type': 'ai', 'author': 'Game Master'})

        mock_user_session_set.assert_called_with("state", state)


@pytest.mark.asyncio
async def test_command_storyboard_enabled(mock_session_data):
    state, _, user_session_get = mock_session_data
    # Add a GM message to the state
    gm_msg = AIMessage(content="Last scene description", name="Game Master", metadata={"message_id": "gm-msg-for-storyboard"})
    state.messages.append(gm_msg)

    with patch("src.commands.cl.user_session.get", side_effect=user_session_get), \
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls, \
         patch("src.commands.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_agent, \
         patch("src.commands.IMAGE_GENERATION_ENABLED", True): # Ensure enabled

        mock_cl_message_instance = AsyncMock()
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
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls, \
         patch("src.commands.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_agent, \
         patch("src.commands.IMAGE_GENERATION_ENABLED", False): # Ensure disabled

        mock_cl_message_instance = AsyncMock()
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
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls, \
         patch("src.commands.storyboard_editor_agent", new_callable=AsyncMock) as mock_storyboard_agent, \
         patch("src.commands.IMAGE_GENERATION_ENABLED", True): # Ensure enabled

        mock_cl_message_instance = AsyncMock()
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
         patch("chainlit.Message", new_callable=AsyncMock) as mock_cl_message_cls:

        mock_cl_message_instance = AsyncMock()
        mock_cl_message_cls.return_value = mock_cl_message_instance

        # Test one command, the logic is the same for all
        await command_roll(query)

        # Check that the "Error: Session state not found." message was sent
        mock_cl_message_cls.assert_called_with(content="Error: Session state not found.")
        mock_cl_message_instance.send.assert_awaited_once_with()
