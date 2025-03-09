import logging
from .state import ChatState
from .initialization import DatabasePool

# Initialize logging
cl_logger = logging.getLogger("chainlit")


async def get_chat_memory(store: DatabasePool) -> ChatState:
    """Get chat memory from store."""
    try:
        state = await store.get_pool().get("chat_state")
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
        # Persist Chainlit thread history separately
        await store.save_chainlit_history(state.thread_id, state.to_dict())
        # Save LangGraph state
        await store.save_langgraph_state(state)
    except Exception as e:
        cl_logger.error(f"Failed to save chat state: {str(e)}", exc_info=True)
        raise
