from typing import List, Annotated, Sequence, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.managed import IsLastStep

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
    """Core state model for the chat application."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    is_last_step: IsLastStep = Field(default_factory=lambda: IsLastStep(False))
    vector_store_id: str | None = None  # Track vector store session
    metadata: Dict[str, Any] = Field(default_factory=dict)  # For additional state info

    def get_recent_history(self, n: int = 5) -> Sequence[BaseMessage]:
        """Get n most recent messages."""
        return self.messages[-n:] if len(self.messages) > n else self.messages

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
