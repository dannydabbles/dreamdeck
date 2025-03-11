from src.config import config
import os
import requests
import logging
from uuid import uuid4  # Import uuid4
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import ToolMessage  # Use LangChain's standard messages
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..config import SERPAPI_KEY, WEB_SEARCH_ENABLED

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _web_search(query: str, store=None, previous=None) -> ToolMessage:
    """Perform a web search using SerpAPI.

    Args:
        query (str): The search query.
        store (BaseStore, optional): The store for chat state. Defaults to None.
        previous (ChatState, optional): Previous chat state. Defaults to None.

    Returns:
        ToolMessage: The search result.
    """
    if not SERPAPI_KEY:
        # Print a warning if the API key is missing
        cl_logger.warning("SerpAPI key is missing.")
    if not WEB_SEARCH_ENABLED:
        return ToolMessage(
            content="Web search is disabled.",
            tool_call_id=str(uuid4()),  # Generate a unique ID for the tool call
            name="error",
        )

    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        response = await requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"Search error: {data['error']}")
        return ToolMessage(
            content=data.get("organic_results", [{}])[0].get("snippet", "No results found."),
            tool_call_id=str(uuid4()),  # Generate a unique ID for the tool call
            name="web_search",
        )
    except requests.exceptions.RequestException as e:
        cl_logger.error(f"Web search failed: {e}")
        return ToolMessage(
            content=f"Web search failed: {str(e)}",
            tool_call_id=str(uuid4()),  # Generate a unique ID for the tool call
            name="error",
        )
    except ValueError as e:
        cl_logger.error(f"Web search failed: {e}")
        return ToolMessage(
            content=f"Web search failed: {str(e)}",
            tool_call_id=str(uuid4()),  # Generate a unique ID for the tool call
            name="error",
        )

@task
async def web_search(query: str, **kwargs) -> ToolMessage:
    return await _web_search(query)

web_search_agent = web_search
