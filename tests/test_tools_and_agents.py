import pytest
from src.state_graph import chat_workflow  # Main entrypoint
from src.agents.dice_agent import dice_roll  # Task function
from src.agents.web_search_agent import web_search  # Task function
from src.config import config
from langgraph.func import task, entrypoint
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from chainlit import Message as CLMessage  # Import CLMessage from Chainlit
from unittest.mock import MagicMock, patch
from uuid import uuid4  # Import uuid4
from chainlit import context as chainlit_context

@pytest.fixture
def mock_chainlit_context():
    from chainlit import context as chainlit_context
    from unittest.mock import MagicMock

    # Mock Chainlit session and context
    mock_session = MagicMock()
    mock_session.id = "test_thread_id"  # Matches what your code expects
    mock_session.user = MagicMock(id="test_user_id", name="Test User")
    mock_session.is_chat = True  # Required for WebSocket context

    mock_context = MagicMock()
    mock_context.session = mock_session
    mock_context.user = mock_session.user
    mock_context.emitter = MagicMock()  # Required for sending messages

    # Initialize the HTTP/WebSocket context with mocks
    chainlit_context.init_http_context(mock_context)
    
    # Ensure the user/session is accessible globally
    with patch("chainlit.context.chainlit_app") as mock_app:
        mock_app.user = mock_session.user
        mock_app.session = mock_session
        
        yield  # Let the test run
    
    # Cleanup after test
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
async def test_decision_agent_roll_action(mock_checkpointer, mock_store):
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
