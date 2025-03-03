import os
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.message import ToolMessage
from .config import WRITER_AGENT_TEMPERATURE, WRITER_AGENT_MAX_TOKENS, WRITER_AGENT_STREAMING, WRITER_AGENT_VERBOSE, LLM_TIMEOUT

# Initialize logging
cl_logger = logging.getLogger("chainlit")

# Initialize the writer AI agent
writer_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=WRITER_AGENT_TEMPERATURE,
        max_tokens=WRITER_AGENT_MAX_TOKENS,
        streaming=WRITER_AGENT_STREAMING,
        verbose=WRITER_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 3,
    ),
    tools=[],
    checkpointer=MemorySaver(),
)
