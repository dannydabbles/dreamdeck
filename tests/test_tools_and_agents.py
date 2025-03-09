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

@pytest.fixture
def mock_checkpointer():
    return MemorySaver()

@pytest.fixture
def mock_runnable_config(mock_checkpointer):
    return {"checkpointer": mock_checkpointer}

@pytest.mark.asyncio
async def test_decision_agent_roll_action(mock_checkpointer):
    # Arrange: Mock the dice_roll task
    with patch('src.agents.dice_agent.dice_roll', return_value=ToolMessage(content="🎲 You rolled 15 on a 20-sided die.", tool_call_id=str(uuid4()), name="dice_roll")):
        # Prepare input message
        user_input = HumanMessage(content="roll 2d20")
        
        # Execute the workflow entrypoint
        state = await chat_workflow([user_input], store={}, previous=None)
        
        # Assert outcome
        assert any(isinstance(msg, ToolMessage) and 'rolled' in msg.content.lower() for msg in state.messages)

@pytest.mark.asyncio
async def test_web_search_integration(mock_checkpointer):
    with patch('src.agents.web_search_agent.web_search', return_value=ToolMessage(content="Web search result: AI is a field of computer science.", tool_call_id=str(uuid4()), name="web_search")):
        user_input = HumanMessage(content="search AI trends")
        state = await chat_workflow([user_input], store={}, previous=None)
        
        # Verify search result inclusion
        assert any('web search result' in msg.content.lower() for msg in state.messages)

@pytest.mark.asyncio
async def test_writer_agent_continuation(mock_checkpointer):
    user_input = HumanMessage(content="Continue the adventure")
    initial_prompt = HumanMessage(content="Previous story context...")
    
    # Execute workflow with mocked state
    state = await chat_workflow([initial_prompt, user_input], store={}, previous=None)
    
    # Ensure AIMessage contains continuation
    ai_responses = [msg for msg in state.messages if isinstance(msg, AIMessage)]
    assert len(ai_responses) >= 1 and ai_responses[-1].content.strip()

@pytest.mark.asyncio
async def test_storyboard_editor_agent(mock_checkpointer):
    user_input = HumanMessage(content="Generate a storyboard")
    initial_prompt = HumanMessage(content="Previous story context...")
    
    # Execute workflow with mocked state
    state = await chat_workflow([initial_prompt, user_input], store={}, previous=None)
    
    # Ensure AIMessage contains storyboard
    ai_responses = [msg for msg in state.messages if isinstance(msg, AIMessage)]
    assert len(ai_responses) >= 1 and ai_responses[-1].content.strip()
