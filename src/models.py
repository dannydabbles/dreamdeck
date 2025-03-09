from typing import List, Annotated, Sequence, Dict, Any, Optional
from typing import ClassVar
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,  # Import ToolMessage from LangGraph
)
from chainlit import Message as CLMessage  # Import Chainlit's UI-facing message
import operator
import chainlit as cl


class MessageType(str, Enum):
    """Enum for message types in the chat."""
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"
    SYSTEM = "system"


class Message(BaseModel):
    """Model for chat messages with rich metadata.

    Attributes:
        content (str): The content of the message.
        type (MessageType): The type of the message.
        metadata (Dict[str, Any], optional): Additional metadata. Defaults to an empty dictionary.
        created_at (datetime, optional): The creation timestamp. Defaults to the current time.
    """

    content: str
    type: MessageType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_langchain_message(self) -> BaseMessage:
        """Convert to LangChain message format.

        Returns:
            BaseMessage: The LangChain message.
        """
        message_map = {
            MessageType.HUMAN: HumanMessage,
            MessageType.AI: AIMessage,
            MessageType.TOOL: ToolMessage,
            MessageType.SYSTEM: SystemMessage,
        }
        return message_map[self.type](
            content=self.content, additional_kwargs=self.metadata
        )


class ChatState(BaseModel):
    """Enhanced state model with Chainlit integration."""
    messages: List[BaseMessage] = []
    is_last_step: bool = False
    thread_id: str = Field(default_factory=lambda: cl.context.session.id)
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
        """Get formatted string of memories.

        Returns:
            str: The formatted string of memories.
        """
        return "\n".join(self.memories) if self.memories else ""

    def get_recent_history_str(self) -> str:
        """Get formatted string of recent message history.

        Returns:
            str: The formatted string of recent message history.
        """
        recent_messages = self.get_recent_history()
        filtered_messages = [
            msg
            for msg in recent_messages
            if isinstance(msg, (HumanMessage, AIMessage))
            or (isinstance(msg, ToolMessage) and "roll" in msg.content.lower())
        ]
        return "\n".join(
            [
                f"{'Human' if isinstance(msg, HumanMessage) else 'GM' if isinstance(msg, AIMessage) else 'Dice'}: {msg.content}"
                for msg in filtered_messages
            ]
        )

    def get_tool_results_str(self) -> str:
        """Get formatted string of recent tool results from message history.

        Returns:
            str: The formatted string of recent tool results.
        """
        tool_results = []
        for msg in self.messages[-10:]:  # Last 10 messages should be enough
            if isinstance(msg, ToolMessage):
                tool_results.append(f"{msg.name}: {msg.content}")
        return "\n".join(tool_results) if tool_results else ""

    def format_system_message(self) -> None:
        """Format the system message with current context."""
        if not self.messages:
            return

        sys_msg = self.messages[0]
        if not isinstance(sys_msg, SystemMessage):
            return

        vector_memory = cl.user_session.get("vector_memory")
        memories = []
        if vector_memory:
            recent_content = "\n".join(msg.content for msg in self.messages[-3:])
            try:
                relevant_docs = vector_memory.get(recent_content)
                memories = [doc.page_content for doc in relevant_docs]
            except Exception as e:
                cl.logger.error(f"Failed to get memories: {e}")

        self.messages[0] = SystemMessage(
            content=sys_msg.content.format(
                recent_chat_history=self.get_recent_history_str(),
                memories="\n".join(memories) if memories else "",
                tool_results=self.get_tool_results_str(),
            )
        )

    def get_recent_history(self, n: int = 5) -> Sequence[BaseMessage]:
        """Get n most recent messages.

        Args:
            n (int, optional): The number of recent messages to retrieve. Defaults to 5.

        Returns:
            Sequence[BaseMessage]: The recent messages.
        """
        return self.messages[-n:] if len(self.messages) > n else self.messages
