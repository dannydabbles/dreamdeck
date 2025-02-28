import logging
from .state import ChatState
from langgraph.store.base import BaseStore
from .initialization import DatabasePool

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def get_chat_memory(store: BaseStore) -> ChatState:
    """Get chat memory from store."""
    try:
        state = await store.get("chat_state")
        if state:
            return ChatState.parse_obj(state)
        else:
            cl_logger.info("No chat state found, initializing new state.")
            return ChatState()
    except Exception as e:
        cl_logger.error(f"Database error: {str(e)}")
        raise

async def save_chat_memory(state: ChatState, store: BaseStore) -> None:
    """Save chat memory to store."""
    try:
        await store.put("chat_state", state.current_message_id, state.dict())
        cl_logger.info(f"Chat state saved successfully: {state.current_message_id}")
    except Exception as e:
        cl_logger.error(f"Failed to save chat state: {str(e)}", exc_info=True)
        raise
