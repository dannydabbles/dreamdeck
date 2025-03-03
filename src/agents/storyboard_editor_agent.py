import os
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.message import ToolMessage
from ..config import STORYBOARD_EDITOR_AGENT_TEMPERATURE, STORYBOARD_EDITOR_AGENT_MAX_TOKENS, STORYBOARD_EDITOR_AGENT_STREAMING, STORYBOARD_EDITOR_AGENT_VERBOSE, LLM_TIMEOUT

# Initialize logging
cl_logger = logging.getLogger("chainlit")

# Initialize the storyboard editor agent
storyboard_editor_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=STORYBOARD_EDITOR_AGENT_TEMPERATURE,
        max_tokens=STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
        streaming=STORYBOARD_EDITOR_AGENT_STREAMING,
        verbose=STORYBOARD_EDITOR_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 2,
    ),
    tools=[],
    checkpointer=MemorySaver(),
)
