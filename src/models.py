from typing import List, Optional
import asyncio
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel, Field, ConfigDict

import chainlit as cl


class ChatState(BaseModel):
    """Represents the conversational state of a user interaction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: List[BaseMessage] = Field(default_factory=list)
    is_last_step: bool = False
    thread_id: str
    metadata: dict = Field(default_factory=dict)
    current_message_id: Optional[str] = None
    error_count: int = 0
    memories: List[str] = Field(default_factory=list)
    user_preferences: dict = Field(default_factory=dict)
    thread_data: dict = Field(default_factory=dict)
    current_persona: str = "Default"  # Default persona if none selected/resumed
    last_agent_called: Optional[str] = None  # Track last agent that added a message
    tool_results_this_turn: List[BaseMessage] = Field(default_factory=list) # Phase 3: Track results within a turn
    background_tasks: List[asyncio.Task] = Field(default_factory=list) # List of background asyncio tasks to track

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
        """Return formatted string of recent tool results, prioritizing current turn."""
        tool_msgs = []
        # Prioritize results from the current turn
        if self.tool_results_this_turn:
            tool_msgs = self.tool_results_this_turn[-3:] # Get last 3 from current turn
        else:
            # Fallback: Look back through message history for last 3 tool messages
            recent_messages = self.messages[-500:]
            tool_msgs = [
                msg for msg in reversed(recent_messages)
                if isinstance(msg, (ToolMessage, AIMessage)) and msg.name in ["dice_roll", "web_search", "todo", "report", "knowledge", "storyboard"]
            ][:3]
        return (
            "\n".join([f"{msg.name}: {msg.content}" for msg in tool_msgs])
            if tool_msgs
            else ""
        )

    def get_recent_history(self, n: int = 500) -> List[BaseMessage]:
        return self.messages[-n:] if len(self.messages) > n else self.messages
