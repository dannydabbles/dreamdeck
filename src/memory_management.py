import logging
from .state import ChatState  # Update import path
from langgraph.store.base import BaseStore

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def get_chat_memory(store: BaseStore) -> ChatState:
    """Get chat memory from store."""
    state = await store.get("chat_state")
    return ChatState.parse_obj(state) if state else ChatState()

async def save_chat_memory(state: ChatState, store: BaseStore) -> None:
    """Save chat memory to store."""
    await store.put("chat_state", state.current_message_id, state.dict())
