from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import BaseMessage

class ChatState(BaseModel):
    """Enhanced state model for the chat.
    
    Attributes:
        messages (List[BaseMessage]): The list of messages.
        metadata (dict, optional): Additional metadata. Defaults to an empty dictionary.
        current_message_id (Optional[str], optional): The current message ID. Defaults to None.
        tool_results (List[str], optional): List of tool results. Defaults to an empty list.
        error_count (int, optional): The error count. Defaults to 0.
    """
    messages: List[BaseMessage] = []
    metadata: dict = {}
    current_message_id: Optional[str] = None
    tool_results: List[str] = []
    error_count: int = 0
    
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
