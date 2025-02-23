from typing import List, Optional, Dict, Any, Sequence
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage

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
    """Enhanced model for managing chat state and memory."""
    messages: List[Message] = Field(default_factory=list)
    current_context: List[Message] = Field(default_factory=list, max_length=10)
    short_term_memory: Dict[str, Any] = Field(default_factory=dict)
    long_term_memory: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, content: str, msg_type: MessageType, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message and maintain context window."""
        message = Message(
            content=content,
            type=msg_type,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.current_context.append(message)
        if len(self.current_context) > 10:
            self.current_context.pop(0)
        return message
            
    def get_recent_history(self, n: int = 5) -> List[Message]:
        """Get n most recent messages."""
        return self.messages[-n:] if len(self.messages) > n else self.messages
    
    def to_langchain_messages(self) -> List[BaseMessage]:
        """Convert all messages to LangChain format."""
        return [msg.to_langchain_message() for msg in self.messages]
    
    def get_context_window(self) -> List[Message]:
        """Get current context window."""
        return self.current_context
    
    def clear_context(self) -> None:
        """Clear current context while preserving history."""
        self.current_context = []

class ToolInput(BaseModel):
    """Model for structured tool inputs."""
    tool_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ToolOutput(BaseModel):
    """Model for structured tool outputs."""
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ImageGenerationState(BaseModel):
    """Model for tracking image generation state."""
    prompts: List[str] = Field(default_factory=list)
    generated_images: List[str] = Field(default_factory=list)
    pending_generations: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_prompt(self, prompt: str) -> None:
        """Add a new image generation prompt."""
        self.prompts.append(prompt)
        self.pending_generations.append(prompt)
    
    def mark_generation_complete(self, prompt: str, image_path: str) -> None:
        """Mark an image generation as complete."""
        if prompt in self.pending_generations:
            self.pending_generations.remove(prompt)
            self.generated_images.append(image_path)
    
    def mark_generation_failed(self, prompt: str, error: str) -> None:
        """Mark an image generation as failed."""
        if prompt in self.pending_generations:
            self.pending_generations.remove(prompt)
            self.errors.append(error)
