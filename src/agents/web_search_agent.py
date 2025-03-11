from src.config import config
import os
import requests
import logging
from uuid import uuid4  # Import uuid4
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # Use LangChain's standard messages
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..config import SERPAPI_KEY, WEB_SEARCH_ENABLED
from ..models import ChatState

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _web_search(state: ChatState) -> list[BaseMessage]:
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
        return AIMessage(
            content="Web search is disabled.",
            name="error",
        )
    
    # Query is last human message
    query = next((m for m in reversed(state.messages) if isinstance(m, HumanMessage)), None)

    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        response = await requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"Search error: {data['error']}")
        return AIMessage(
            content=data.get("organic_results", [{}])[0].get("snippet", "No results found."),
            name="web_search",
        )
    except requests.exceptions.RequestException as e:
        cl_logger.error(f"Web search failed: {e}")
        return AIMessage(
            content=f"Web search failed: {str(e)}",
            name="error",
        )
    except ValueError as e:
        cl_logger.error(f"Web search failed: {e}")
        return AIMessage(
            content=f"Web search failed: {str(e)}",
            name="error",
        )

@task
async def web_search(state: ChatState) -> list[BaseMessage]:
    return await _web_search(state)

web_search_agent = web_search
