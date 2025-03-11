from typing import List, Optional
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from pydantic import BaseModel

class ChatState(BaseModel):
    messages: List[BaseMessage] = []
    is_last_step: bool = False
    thread_id: str = ...
    metadata: dict = {}
    current_message_id: Optional[str] = None
    tool_results: List[str] = []
    error_count: int = 0
    memories: List[str] = []
    user_preferences: dict = {}
    thread_data: dict = {}

    def add_tool_result(self, result: str) -> None:
        self.tool_results.append(result)

    def clear_tool_results(self) -> None:
        self.tool_results = []

    def increment_error_count(self) -> None:
        self.error_count += 1

    def get_memories_str(self) -> str:
        return "\n".join(self.memories) if self.memories else ""

    def get_recent_history_str(self) -> str:
        recent_messages = self.messages[-5:]
        filtered = [
            msg 
            for msg in recent_messages 
            if isinstance(msg, (HumanMessage, AIMessage)) 
            or (isinstance(msg, ToolMessage) and "roll" in msg.content.lower())
        ]
        return "\n".join([
            f"{'Human' if isinstance(msg, HumanMessage) else 'GM' if isinstance(msg, AIMessage) else 'Dice'}: {msg.content}"
            for msg in filtered
        ])

    def get_tool_results_str(self) -> str:
        tool_msgs = [msg for msg in self.messages[-10:] if isinstance(msg, ToolMessage)]
        return "\n".join([f"{msg.name}: {msg.content}" for msg in tool_msgs]) if tool_msgs else ""

    def add_tool_message(self, tool_message: ToolMessage) -> None:
        self.messages.append(tool_message)

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
