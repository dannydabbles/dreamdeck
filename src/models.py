from typing import List, Annotated, Sequence, Dict, Any, Optional
from pydantic import BaseModel, Field
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
    """Model for chat messages with rich metadata."""
    content: str
    type: MessageType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    def to_langchain_message(self) -> BaseMessage:
        """Convert to LangChain message format."""
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
    """Enhanced state model with Chainlit integration."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    is_last_step: IsLastStep = Field(default_factory=lambda: IsLastStep(False))
    thread_id: str = Field(default_factory=lambda: cl.context.session.id)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_tool: str | None = None
    tool_results: List[str] = Field(default_factory=list)
    error_count: int = Field(default=0)

    def get_thread_context(self) -> Dict[str, Any]:
        """Get context from thread history."""
        thread = cl.user_session.get("thread")
        if not thread:
            return {}
        return {
            "thread_id": thread.get("id"),
            "user_id": thread.get("userId"),
            "created_at": thread.get("createdAt")
        }

    def to_chainlit_message(self, message: BaseMessage) -> Dict[str, Any]:
        """Convert message to Chainlit format."""
        return {
            "content": message.content,
            "type": "ai_message" if isinstance(message, AIMessage) else "user_message",
            "metadata": message.additional_kwargs
        }

    def get_recent_history(self, n: int = 5) -> Sequence[BaseMessage]:
        """Get n most recent messages."""
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def add_tool_result(self, result: str) -> None:
        """Add a tool result to the state."""
        self.tool_results.append(result)
    
    def clear_tool_results(self) -> None:
        """Clear tool results after processing."""
        self.tool_results = []
    
    def increment_error_count(self) -> None:
        """Increment error count for retry logic."""
        self.error_count += 1

class ImageGenerationState(BaseModel):
    """State model for image generation."""
    prompts: List[str] = Field(default_factory=list)
    parent_message_id: str | None = None
    status: str = "pending"
    generated_images: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_prompt(self, prompt: str) -> None:
        """Add a new image generation prompt."""
        self.prompts.append(prompt)
        self.status = "pending"
    
    def mark_generation_complete(self, image_path: str) -> None:
        """Mark an image generation as complete."""
        self.generated_images.append(image_path)
        if len(self.generated_images) == len(self.prompts):
            self.status = "complete"
    
    def mark_generation_failed(self, error: str) -> None:
        """Mark an image generation as failed."""
        self.errors.append(error)
        self.status = "failed"
