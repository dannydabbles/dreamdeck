from typing import List, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

import chainlit as cl


class ChatState(BaseModel):
    messages: List[BaseMessage] = []
    is_last_step: bool = False
    thread_id: str = ...
    metadata: dict = {}
    current_message_id: Optional[str] = None
    error_count: int = 0
    memories: List[str] = []
    user_preferences: dict = {}
    thread_data: dict = {}

    def increment_error_count(self) -> None:
        self.error_count += 1

    def get_memories_str(self) -> str:
        return "\n".join(self.memories) if self.memories else ""

    def get_recent_history_str(self) -> str:
        # TODO: Summarize messages past cutoff
        recent_messages = self.messages[-500:]
        filtered = [
            msg 
            for msg in recent_messages 
            if isinstance(msg, (HumanMessage, AIMessage))
        ]
        return "\n".join([
            f"{msg.name}: {msg.content}"
            for msg in filtered
        ])

    def get_tool_results_str(self) -> str:
        recent_messages = self.messages[-5:]
        # Go through recent messages and find the last AIMessage with name in ["dice_roll", "web_search"]
        recent_agent_messages = [
            msg for msg in reversed(recent_messages) if isinstance(msg, AIMessage) and msg.name in ["dice_roll", "web_search"]
        ]
        # Set tool_msgs to the last agent message if it exists
        tool_msgs = recent_agent_messages[:1]
        return "\n".join([f"{msg.name}: {msg.content}" for msg in tool_msgs]) if tool_msgs else ""

    def format_system_message(self) -> None:
        vector_memory = cl.user_session.get("vector_memory")
        if vector_memory:
            recent_content = "\n".join(msg.content for msg in self.messages[-3:])
            try:
                relevant_docs = vector_memory.get(recent_content)
                memories = [doc.page_content for doc in relevant_docs]
            except Exception as e:
                cl.logger.error(f"Failed to get memories: {e}")

        if self.messages and isinstance(self.messages[0], SystemMessage):
            self.messages[0] = SystemMessage(
                content=self.messages[0].content.format(
                    recent_chat_history=self.get_recent_history_str(),
                    memories="\n".join(memories) if memories else "",
                    tool_results=self.get_tool_results_str(),
                )
            )

    def get_recent_history(self, n: int = 5) -> List[BaseMessage]:
        return self.messages[-n:] if len(self.messages) > n else self.messages
