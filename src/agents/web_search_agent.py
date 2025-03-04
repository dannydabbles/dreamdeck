import os
import requests
import logging
from langgraph.prebuilt import create_react_agent
from typing import Dict
from ..config import SERPAPI_KEY, WEB_SEARCH_ENABLED
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver

# Initialize logging
cl_logger = logging.getLogger("chainlit")

def web_search(query: str) -> ToolMessage:
    """Perform a web search using SerpAPI.

    Args:
        query (str): The search query.

    Returns:
        ToolMessage: The search result.
    """
    if not SERPAPI_KEY:
        return {"name": "error", "args": {"content": "SERPAPI_KEY environment variable not set."}}
    if not WEB_SEARCH_ENABLED:
        return {"name": "error", "args": {"content": "Web search is disabled."}}

    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"Search error: {data['error']}")
        return {"name": "web_search", "args": {"result": data.get("organic_results", [{}])[0].get("snippet", "No results found.")}}
    except requests.exceptions.RequestException as e:
        cl_logger.error(f"Web search failed: {e}", exc_info=True)
        return {"name": "error", "args": {"content": f"Web search failed: {str(e)}"}}
    except ValueError as e:
        cl_logger.error(f"Web search failed: {e}", exc_info=True)
        return ToolMessage(content=f"Web search failed: {str(e)}")

from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver

# Initialize the web search agent
web_search_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=0.0,
        max_tokens=100,
        streaming=False,
        verbose=False,
        request_timeout=os.getenv("LLM_TIMEOUT", 10),
    ),
    tools=[web_search],
    checkpointer=MemorySaver(),
)
