import random
from typing import Optional, List
import requests  # Import requests
import os  # Import os

from langgraph.prebuilt import ToolNode, ToolExecutor
from langchain.prompts import PromptTemplate  # Import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.schema.output_parser import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import logging

# Initialize logging
cl_logger = logging.getLogger("chainlit")

from config import DICE_SIDES, LLM_TEMPERATURE, LLM_STREAMING, LLM_MODEL_NAME, LLM_TIMEOUT, LLM_MAX_TOKENS, LLM_PRESENCE_PENALTY, LLM_FREQUENCY_PENALTY, LLM_TOP_P, LLM_VERBOSE, AI_WRITER_PROMPT  # Import constants

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

def categorize_input(input: str) -> str:
    """
    Categorizes the user input into one of the following categories: 'roll', 'search', 'continue_story'.

    Args:
        input (str): The user input.

    Returns:
        str: The category of the input.
    """
    input = input.lower()
    if "roll" in input:
        return "roll"
    if "search" in input or "look up" in input or "find out" in input:
        return "search"
    return "continue_story"

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

# Initialize the image generation AI agent
storyboard_model = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    temperature=LLM_TEMPERATURE,
    streaming=False,
    model_name=LLM_MODEL_NAME,
    request_timeout=LLM_TIMEOUT,
    max_tokens=LLM_MAX_TOKENS,
    presence_penalty=LLM_PRESENCE_PENALTY,
    frequency_penalty=LLM_FREQUENCY_PENALTY,
    top_p=LLM_TOP_P,
    verbose=LLM_VERBOSE
)

storyboard_editor_agent = storyboard_model
