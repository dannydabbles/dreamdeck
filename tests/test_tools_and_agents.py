import pytest
from unittest.mock import MagicMock
from chainlit import context as chainlit_context
from chainlit.types import ChainlitContext, Session

@pytest.fixture
def mock_chainlit_context():
    # Create mock session and context
    mock_session = MagicMock(spec=Session)
    mock_session.id = "test_thread_id"
    mock_session.user = MagicMock(id="test_user_id", name="Test User")
    mock_session.is_chat = True

    mock_emitter = MagicMock()
    mock_context = ChainlitContext(
        session=mock_session,
        emitter=mock_emitter,
        user=mock_session.user,
    )
    mock_context.chat = MagicMock()
    mock_context.chat_context = MagicMock()

    # Initialize the HTTP context
    chainlit_context.init_http_context(mock_context)
    yield mock_context  # Let the test run

    # Cleanup (if needed)
    chainlit_context.reset()

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
        config = {"configurable": {"thread_id": "test_thread"}}
        state = await chat_workflow.ainvoke(input=input_data, config=config)
        
        # Assert outcome
        assert any(isinstance(msg, ToolMessage) and 'rolled' in msg.content.lower() for msg in state.messages)
        assert mock_store.put.called  # VERIFY STORE USAGE

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_chainlit_context")
async def test_web_search_integration(mock_checkpointer):
    with patch('src.agents.web_search_agent.web_search', return_value=ToolMessage(content="Web search result: AI is a field of computer science.", tool_call_id=str(uuid4()), name="web_search")), \
         patch('src.event_handlers.cl_user_session.get', return_value={}):
        user_input = HumanMessage(content="search AI trends")
        input_data = {
            "messages": [user_input],
            "store": {},
            "previous": None
        }
        config = {"configurable": {"thread_id": "test_thread"}}
        state = await chat_workflow.ainvoke(input=input_data, config=config)
        
        # Verify search result inclusion
        assert any('web search result' in msg.content.lower() for msg in state.messages)

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_chainlit_context")
async def test_writer_agent_continuation(mock_checkpointer):
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
        state = await chat_workflow.ainvoke(input=input_data, config=config)
        
        # Ensure AIMessage contains continuation
        ai_responses = [msg for msg in state.messages if isinstance(msg, AIMessage)]
        assert len(ai_responses) >= 1 and ai_responses[-1].content.strip()

@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_chainlit_context")
async def test_storyboard_editor_agent(mock_checkpointer):
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
        state = await chat_workflow.ainvoke(input=input_data, config=config)
        
        # Ensure AIMessage contains storyboard
        ai_responses = [msg for msg in state.messages if isinstance(msg, AIMessage)]
        assert len(ai_responses) >= 1 and ai_responses[-1].content.strip()
