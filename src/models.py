from typing import List, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field

import chainlit as cl


class ChatState(BaseModel):
    """Represents the conversational state of a user interaction.

    Attributes:
        messages (List[BaseMessage]): Conversation history including user/system messages
        is_last_step (bool): Indicates if this is terminal state (unused currently)
        thread_id (str): Unique identifier for the conversation thread
        metadata (dict): Arbitrary state data for plugins/extensions
        current_message_id (Optional[str]): Reference to ongoing message (async ops)
        error_count (int): Number of consecutive errors encountered
        memories (List[str]): Extracted contextual info from prior messages
        user_preferences (dict): User-specific settings like tone preference
        thread_data (dict): Metadata about the conversation thread
        current_persona (str): The active persona/profile name.
        last_agent_called (Optional[str]): The last agent that successfully added a message.

    State Transitions:
        - Messages are immutable - new states are created with .copy_with_updates()
        - Memories are automatically populated from vector store queries
    """

    messages: List[BaseMessage] = Field(default_factory=list)
    is_last_step: bool = False
    thread_id: str
    metadata: dict = Field(default_factory=dict)
    current_message_id: Optional[str] = None
    error_count: int = 0
    memories: List[str] = Field(default_factory=list)
    user_preferences: dict = Field(default_factory=dict)
    thread_data: dict = Field(default_factory=dict)
    current_persona: str = "Default" # Default persona if none selected/resumed
    last_agent_called: Optional[str] = None  # Track last agent that added a message

    def increment_error_count(self) -> None:
        self.error_count += 1

    def get_memories_str(self) -> str:
        try:
            return "\n".join([doc for doc in self.memories[:3]])
        except Exception:
            return ""

    def get_last_human_message(self) -> Optional[HumanMessage]:
        human_messages = [
            msg for msg in reversed(self.messages) if isinstance(msg, HumanMessage)
        ]
        return human_messages[0] if human_messages else None

    def get_recent_history_str(self, n: int = 500) -> str:
        """Return last N messages as formatted strings, excluding non-chat messages."""
        recent_messages = self.messages[-n:] if self.messages else []
        filtered = [
            msg for msg in recent_messages if isinstance(msg, (HumanMessage, AIMessage))
        ]
        return "\n".join([f"{msg.name}: {msg.content}" for msg in filtered])

    def get_tool_results_str(self) -> str:
        recent_messages = self.messages[-500:]
        # Go through recent messages and find the last AIMessage with name in ["dice_roll", "web_search"]
        recent_agent_messages = [
            msg
            for msg in reversed(recent_messages)
            if isinstance(msg, AIMessage) and msg.name in ["dice_roll", "web_search"]
        ]
        # Set tool_msgs to the last 3 agent messages if they exist
        tool_msgs = recent_agent_messages[:3]
        return (
            "\n".join([f"{msg.name}: {msg.content}" for msg in tool_msgs])
            if tool_msgs
            else ""
        )

    def get_recent_history(self, n: int = 500) -> List[BaseMessage]:
        return self.messages[-n:] if len(self.messages) > n else self.messages
