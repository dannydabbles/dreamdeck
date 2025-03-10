import pytest
from unittest.mock import MagicMock, AsyncMock
from chainlit import context, user_session, message
from chainlit.types import ChainlitRequest
from chainlit.sdk import ChainlitSdk

@pytest.fixture
def mock_chainlit_context():
    from chainlit import context, user_session as cl_user_session, message as cl_message

    context.cycle()  # Start fresh context
    context.session = MagicMock(id="test-session-id")
    context.emitter = MagicMock()
    cl_user_session.set("state", ChatState())
    cl_user_session.set("vector_memory", MagicMock())

    # Mock message sending
    cl_message.Message = MagicMock()
    cl_message.Message.send = AsyncMock()

    # Mock vector store and database connections
    cl_user_session.set("vector_memory", MagicMock())
    cl_user_session.set("database", MagicMock())

    # Mock LangGraph store
    from langgraph.checkpoint.memory import MemorySaver
    cl_user_session.set("checkpointer", MemorySaver())

    yield context

    context.reset()

@pytest.fixture
def mock_checkpointer():
    return MemorySaver()

@pytest.fixture
def mock_store():
    class MockStore:
        def put(self, *args): pass
        def get(self, *args): return {}
    return MockStore()

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_chainlit_context")
async def test_decision_agent_roll_action(mock_checkpointer, mock_store, mock_chainlit_context):
    with patch('src.agents.dice_agent.dice_roll', return_value=ToolMessage(content="🎲 You rolled 15 on a 20-sided die.", tool_call_id=str(uuid4()), name="dice_roll")), \
         patch('src.event_handlers.cl_user_session.get', return_value={}):
        # Prepare input message
        user_input = HumanMessage(content="roll 2d20")
        
        # Execute the workflow entrypoint
        input_data = {
            "messages": [user_input],
            "store": mock_store,
            "previous": None
        }
        config = {"configurable": {"thread_id": "test-thread-id"}}
        state = await chat_workflow.invoke(input_data, config=config)
        
        # Assert outcome
        assert any(isinstance(msg, ToolMessage) and 'rolled' in msg.content.lower() for msg in state.messages)
        assert await mock_store.put.called  # VERIFY STORE USAGE

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_chainlit_context")
async def test_web_search_integration(mock_checkpointer, mock_chainlit_context):
    with patch('src.agents.web_search_agent.web_search', return_value=ToolMessage(content="Web search result: AI is a field of computer science.", tool_call_id=str(uuid4()), name="web_search")), \
         patch('src.event_handlers.cl_user_session.get', return_value={}):
        user_input = HumanMessage(content="search AI trends")
        input_data = {
            "messages": [user_input],
            "store": {},
            "previous": None
        }
        config = {"configurable": {"thread_id": "test_thread"}}
        state = await chat_workflow.invoke(input_data, config=config)
        
        # Verify search result inclusion
        assert any('web search result' in msg.content.lower() for msg in state.messages)

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_chainlit_context")
async def test_writer_agent_continuation(mock_checkpointer, mock_chainlit_context):
    with patch('src.event_handlers.cl_user_session.get', return_value={}):
        user_input = HumanMessage(content="Continue the adventure")
        initial_prompt = HumanMessage(content="Previous story context...")
        
        # Execute workflow with mocked state
        input_data = {
            "messages": [initial_prompt, user_input],
            "store": {},
            "previous": None
        }
        config = {"configurable": {"thread_id": "test_thread"}}
        state = await chat_workflow.invoke(input_data, config=config)
        
        # Ensure AIMessage contains continuation
        ai_responses = [msg for msg in state.messages if isinstance(msg, AIMessage)]
        assert len(ai_responses) >= 1 and ai_responses[-1].content.strip()

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_chainlit_context")
async def test_storyboard_editor_agent(mock_checkpointer, mock_chainlit_context):
    with patch('src.event_handlers.cl_user_session.get', return_value={}):
        user_input = HumanMessage(content="Generate a storyboard")
        initial_prompt = HumanMessage(content="Previous story context...")
        
        # Execute workflow with mocked state
        input_data = {
            "messages": [initial_prompt, user_input],
            "store": {},
            "previous": None
        }
        config = {"configurable": {"thread_id": "test_thread"}}
        state = await chat_workflow.invoke(input_data, config=config)
        
        # Ensure AIMessage contains storyboard
        ai_responses = [msg for msg in state.messages if isinstance(msg, AIMessage)]
        assert len(ai_responses) >= 1 and ai_responses[-1].content.strip()
