import logging
from .state import ChatState  # Update import path
from langgraph.store.base import BaseStore
from src.initialization import DatabasePool

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def get_chat_memory(store: BaseStore) -> ChatState:
    """Get chat memory from store."""
    try:
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as connection:
            # Perform your query here
            state = await store.get("chat_state")
            return ChatState.parse_obj(state) if state else ChatState()
    except Exception as e:
        cl_logger.error(f"Database error: {str(e)}")
        raise

async def save_chat_memory(state: ChatState, store: BaseStore) -> None:
    """Save chat memory to store."""
    try:
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as connection:
            await store.put("chat_state", state.current_message_id, state.dict())
    except Exception as e:
        cl_logger.error(f"Database error: {str(e)}")
        raise
