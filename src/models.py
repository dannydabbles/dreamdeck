from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class MessageType(str, Enum):
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"
    TOOL = "tool"

class Message(BaseModel):
    content: str
    type: MessageType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

class ChatState(BaseModel):
    messages: List[Message] = Field(default_factory=list)
    current_context: List[Message] = Field(default_factory=list)
    memory: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, content: str, msg_type: MessageType, metadata: Optional[Dict[str, Any]] = None):
        message = Message(
            content=content,
            type=msg_type,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.current_context.append(message)
        if len(self.current_context) > 10:  # Keep last 10 messages for context
            self.current_context.pop(0)
            
    def get_recent_history(self, n: int = 5) -> List[Message]:
        return self.messages[-n:] if len(self.messages) > n else self.messages
