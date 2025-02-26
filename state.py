from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import BaseMessage

class ChatState(BaseModel):
    messages: List[BaseMessage] = []
    metadata: dict = {}
    current_message_id: Optional[str] = None
    tool_results: List[str] = []
    error_count: int = 0
    
    def add_tool_result(self, result: str) -> None:
        self.tool_results.append(result)
    
    def clear_tool_results(self) -> None:
        self.tool_results = []
    
    def increment_error_count(self) -> None:
        self.error_count += 1
