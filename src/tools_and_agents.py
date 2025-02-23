import os
import random
import requests
from typing import Optional, List
from langgraph.prebuilt import ToolNode, ToolExecutor
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import logging

# Initialize logging
cl_logger = logging.getLogger("chainlit")

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

# Initialize the decision agent with proper configuration
decision_agent = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    temperature=0.2,  # Lower temperature for more consistent decisions
    streaming=False,
    model_name=LLM_MODEL_NAME,
    request_timeout=LLM_TIMEOUT,
    max_tokens=100,  # Keep responses short for decisions
    verbose=LLM_VERBOSE
).with_structured_output(str)  # Ensure string output

# Initialize the writer AI agent
writer_model = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    temperature=LLM_TEMPERATURE,
    streaming=LLM_STREAMING,
    model_name=LLM_MODEL_NAME,
    request_timeout=LLM_TIMEOUT,
    max_tokens=LLM_MAX_TOKENS,
    presence_penalty=LLM_PRESENCE_PENALTY,
    frequency_penalty=LLM_FREQUENCY_PENALTY,
    top_p=LLM_TOP_P,
    verbose=LLM_VERBOSE
)

writer_prompt = PromptTemplate.from_template(AI_WRITER_PROMPT)

# Assuming ChatOpenAI has a bind_tools method; if not, adjust accordingly
writer_agent = writer_model.bind_tools([dice_roll, web_search])

# Initialize the storyboard editor agent with proper configuration
storyboard_editor_agent = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    temperature=0.7,  # Slightly higher temperature for creative generation
    streaming=False,
    model_name=LLM_MODEL_NAME,
    request_timeout=LLM_TIMEOUT,
    max_tokens=LLM_MAX_TOKENS,
    presence_penalty=0.1,
    frequency_penalty=0.1,
    top_p=0.9,
    verbose=LLM_VERBOSE
)
