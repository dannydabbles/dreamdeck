from typing import List, Annotated, Sequence, Dict, Any, Optional
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.managed import IsLastStep
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
            MessageType.SYSTEM: SystemMessage
        }
        return message_map[self.type](
            content=self.content,
            additional_kwargs=self.metadata
        )

class ChatState(BaseModel):
    """Enhanced state model with Chainlit integration.
    
    Attributes:
        messages (Sequence[BaseMessage]): The list of messages.
        is_last_step (IsLastStep, optional): Flag indicating if this is the last step. Defaults to False.
        thread_id (str, optional): The thread ID. Defaults to the current session ID.
        metadata (Dict[str, Any], optional): Additional metadata. Defaults to an empty dictionary.
        current_message_id (Optional[str], optional): The current message ID. Defaults to None.
        tool_results (List[str], optional): List of tool results. Defaults to an empty list.
        error_count (int, optional): The error count. Defaults to 0.
        memories (List[str], optional): List of memories. Defaults to an empty list.
        user_preferences (Dict[str, Any], optional): User preferences. Defaults to an empty dictionary.
        thread_data (Dict[str, Any], optional): Thread-specific data. Defaults to an empty dictionary.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    is_last_step: IsLastStep = Field(default_factory=lambda: IsLastStep(False))
    thread_id: str = Field(default_factory=lambda: cl.context.session.id)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_message_id: Optional[str] = None
    tool_results: List[str] = Field(default_factory=list)
    error_count: int = Field(default=0)
    memories: List[str] = Field(default_factory=list)  # Add memories field
    user_preferences: Dict[str, Any] = Field(default_factory=dict)  # Add user preferences
    thread_data: Dict[str, Any] = Field(default_factory=dict)  # Add thread-specific data

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
            msg for msg in recent_messages 
            if isinstance(msg, (HumanMessage, AIMessage)) or
            (isinstance(msg, ToolMessage) and "roll" in msg.content.lower())
        ]
        return "\n".join([
            f"{'Human' if isinstance(msg, HumanMessage) else 'GM' if isinstance(msg, AIMessage) else 'Dice'}: {msg.content}" 
            for msg in filtered_messages
        ])

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
                tool_results=self.get_tool_results_str()
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
        
    def add_tool_result(self, result: str) -> None:
        """Add a tool result to the state.
        
        Args:
            result (str): The tool result.
        """
        self.tool_results.append(result)
    
    def clear_tool_results(self) -> None:
        """Clear tool results after processing."""
        self.tool_results = []
    
    def increment_error_count(self) -> None:
        """Increment error count for retry logic."""
        self.error_count += 1
