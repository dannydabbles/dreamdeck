import os
import logging
from chainlit import on_chat_start, on_chat_resume, on_message, user_session as cl_user_session  # Import cl_user_session
from chainlit.types import ThreadDict
from chainlit import Message as CLMessage  # Import CLMessage
from .state_graph import chat_workflow
from .memory_management import get_chat_memory, save_chat_memory
from .models import ChatState
from .config import AI_WRITER_PROMPT, CHAINLIT_STARTERS
from .stores import VectorStore
from .tools_and_agents import writer_agent, storyboard_editor_agent, dice_roll, web_search, decision_agent
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # Import SystemMessage and AIMessage

# Initialize logging
cl_logger = logging.getLogger("chainlit")

@on_chat_start
async def on_chat_start():
    """Initialize new chat session with Chainlit integration."""
    state = ChatState(messages=[SystemMessage(content=AI_WRITER_PROMPT)])
    cl_user_session.set("state", state)  # Use cl_user_session
    cl_user_session.set("vector_memory", VectorStore())  # Use cl_user_session
    cl_user_session.set("runnable", chat_workflow)  # Use cl_user_session

    # Send starters
    for starter in CHAINLIT_STARTERS:
        msg = await CLMessage(content=starter).send()
        state.messages.append(AIMessage(content=starter, additional_kwargs={"message_id": msg.id}))

@on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread."""
    state = await get_chat_memory(cl_user_session.get("vector_memory"))  # Use cl_user_session
    cl_user_session.set("state", state)  # Use cl_user_session
    cl_user_session.set("runnable", chat_workflow)  # Use cl_user_session

@on_message
async def on_message(message: CLMessage):
    """Handle incoming messages."""
    state = cl_user_session.get("state")  # Use cl_user_session
    runnable = cl_user_session.get("runnable")  # Use cl_user_session

    if message.type != "user_message":
        return

    try:
        # Add user message to state
        state.messages.append(HumanMessage(content=message.content))

        # Generate AI response
        ai_response = await runnable(state.messages, cl_user_session.get("vector_memory"), state)  # Use cl_user_session
        state.current_message_id = message.id

        # Save state
        await save_chat_memory(state, cl_user_session.get("vector_memory"))  # Use cl_user_session
    except Exception as e:
        cl_logger.error(f"Runnable stream failed: {e}")
        await CLMessage(content="⚠️ An error occurred while generating the response. Please try again later.").send()
