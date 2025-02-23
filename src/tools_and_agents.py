import os
import random
import requests
from typing import Optional, List, Literal
from langgraph.prebuilt import ToolNode, ToolExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class DecisionOutput(BaseModel):
    """Schema for the decision output."""
    action: Literal["roll", "search", "continue_story"] = Field(
        description="The next action to take based on user input"
    )
import logging

# Initialize logging
cl_logger = logging.getLogger("chainlit")

class DecisionOutput(BaseModel):
    """Schema for the decision output."""
    action: Literal["roll", "search", "continue_story"] = Field(
        description="The next action to take based on user input"
    )

from config import (
    DICE_SIDES,
    LLM_TEMPERATURE,
    LLM_STREAMING,
    LLM_MODEL_NAME,
    LLM_TIMEOUT,
    LLM_PRESENCE_PENALTY,
    LLM_FREQUENCY_PENALTY,
    LLM_TOP_P,
    LLM_VERBOSE,
    DECISION_PROMPT,
    AI_WRITER_PROMPT,
    LLM_MAX_TOKENS  # Import LLM_MAX_TOKENS
)

@tool
def dice_roll(n: Optional[int] = DICE_SIDES) -> str:
    """
    Rolls a dice with a specified number of sides.

    Args:
        n (Optional[int]): Number of sides on the dice. Defaults to DICE_SIDES.

    Returns:
        str: Result of the dice roll.
    """
    sides = n if n else DICE_SIDES
    result = random.randint(1, sides)
    cl_logger.info(f"Dice roll result: {result} on a {sides}-sided die.")
    return f"🎲 You rolled a {result} on a {sides}-sided die."

@tool
def web_search(query: str) -> str:
    """
    Performs a web search using SerpAPI.

    Args:
        query (str): The search query.

    Returns:
        str: The search result.
    """
    serpapi_key = os.getenv("SERPAPI_KEY")
    if not serpapi_key:
        raise ValueError("SERPAPI_KEY environment variable not set.")
    url = f"https://serpapi.com/search.json?q={query}&api_key={serpapi_key}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        raise ValueError(f"Search error: {data['error']}")
    return data.get("organic_results", [{}])[0].get("snippet", "No results found.")

from langgraph.prebuilt import ToolNode, ToolExecutor

# Create a parser for the decision output
decision_parser = PydanticOutputParser(pydantic_object=DecisionOutput)

def log_decision_agent_response(response):
    """Log detailed information about the decision agent's response."""
    cl_logger.debug(f"Decision agent raw response: {response}")
    cl_logger.debug(f"Response type: {type(response)}")
    cl_logger.debug(f"Response attributes: {dir(response)}")
    if hasattr(response, 'additional_kwargs'):
        cl_logger.debug(f"Additional kwargs: {response.additional_kwargs}")
    if hasattr(response, 'content'):
        cl_logger.debug(f"Content: {response.content}")

# Initialize the decision agent with proper function binding and longer timeout
decision_agent = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    temperature=0.2,
    streaming=False,
    model_name=LLM_MODEL_NAME,
    request_timeout=LLM_TIMEOUT * 2,  # Double the timeout for decision making
    max_tokens=100,
    verbose=LLM_VERBOSE
).bind(
    function_call={"name": "decide_action"},  # Force this function to be called
    functions=[{
        "name": "decide_action",
        "description": "Decide the next action based on user input",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["roll", "search", "continue_story"],
                    "description": "The action to take"
                }
            },
            "required": ["action"]
        }
    }]
)

# Create tools list and executor
tools = [dice_roll, web_search]
tool_executor = ToolExecutor(tools)

# Create tool node with the tools list, not the executor
tool_node = ToolNode(tools=tools)  # Pass the tools list directly, not the executor

# Create tool schemas that OpenAI can understand
tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "dice_roll",
            "description": "Rolls a dice with a specified number of sides",
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of sides on the dice"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Performs a web search using SerpAPI",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# Initialize the writer AI agent with tools and longer timeout
writer_agent = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    model_name=LLM_MODEL_NAME,
    temperature=LLM_TEMPERATURE,
    max_tokens=LLM_MAX_TOKENS,
    streaming=True,
    request_timeout=LLM_TIMEOUT * 3,
    presence_penalty=LLM_PRESENCE_PENALTY,
    frequency_penalty=LLM_FREQUENCY_PENALTY,
    top_p=LLM_TOP_P,
    verbose=LLM_VERBOSE
).bind(
    tools=[dice_roll, web_search],
    tool_choice="auto",
    tool_schemas=tool_schemas,
    config={
        "run_name": "writer_agent",
        "callbacks": None,
        "tags": ["writer"],
        "metadata": {"type": "writer"}
    }
)

# Initialize the storyboard editor agent with longer timeout
storyboard_editor_agent = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    model_name=LLM_MODEL_NAME,
    temperature=0.7,
    streaming=False,
    request_timeout=LLM_TIMEOUT * 2,
    max_tokens=LLM_MAX_TOKENS,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    top_p=0.9,
    verbose=LLM_VERBOSE
)
