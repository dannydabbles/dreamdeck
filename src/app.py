import os
import logging
from chainlit import on_chat_start, on_chat_resume, on_message
from chainlit.types import ThreadDict
from .state_graph import chat_workflow
from .memory_management import get_chat_memory, save_chat_memory
from .models import ChatState
from .config import AI_WRITER_PROMPT, CHAINLIT_STARTERS
from .stores import VectorStore
from .tools_and_agents import writer_agent, storyboard_editor_agent, dice_roll, web_search, decision_agent

# Initialize logging
cl_logger = logging.getLogger("chainlit")

@on_chat_start
async def on_chat_start():
    """Initialize new chat session with Chainlit integration."""
    state = ChatState(messages=[SystemMessage(content=AI_WRITER_PROMPT)])
    cl.user_session.set("state", state)
    cl.user_session.set("vector_memory", VectorStore())
    cl.user_session.set("runnable", chat_workflow)

    # Send starters
    for starter in CHAINLIT_STARTERS:
        msg = await CLMessage(content=starter).send()
        state.messages.append(AIMessage(content=starter, additional_kwargs={"message_id": msg.id}))

@on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread."""
    state = await get_chat_memory(cl.user_session.get("vector_memory"))
    cl.user_session.set("state", state)
    cl.user_session.set("runnable", chat_workflow)

@on_message
async def on_message(message: CLMessage):
    """Handle incoming messages."""
    state = cl.user_session.get("state")
    runnable = cl.user_session.get("runnable")

    if message.type != "user_message":
        return

    try:
        # Add user message to state
        state.messages.append(HumanMessage(content=message.content))

        # Generate AI response
        ai_response = await runnable(state.messages, cl.user_session.get("vector_memory"), state)
        state.current_message_id = message.id

        # Save state
        await save_chat_memory(state, cl.user_session.get("vector_memory"))
    except Exception as e:
        cl_logger.error(f"Runnable stream failed: {e}")
        await CLMessage(content="⚠️ An error occurred while generating the response. Please try again later.").send()
