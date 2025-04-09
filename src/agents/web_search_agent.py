from src.config import config
import os
import requests
import logging
import urllib.parse
from jinja2 import Template
from uuid import uuid4  # Import uuid4
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    BaseMessage,
)  # Use LangChain's standard messages
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from src.config import SERPAPI_KEY, WEB_SEARCH_ENABLED
from src.models import ChatState

import chainlit as cl

# Initialize logging
cl_logger = logging.getLogger("chainlit")


@cl.step(name="Web Search Agent", type="tool")
async def _web_search(state: ChatState) -> list[BaseMessage]:
    """Generate a search query, call SerpAPI, and summarize results."""
    if not WEB_SEARCH_ENABLED:
        return [AIMessage(content="Web search is disabled.", name="error", metadata={"message_id": None})]

    if not SERPAPI_KEY:
        return [AIMessage(content="SerpAPI key is missing.", name="error", metadata={"message_id": None})]

    # Get user input and context
    user_query = state.get_last_human_message()
    if not user_query:
        return [AIMessage(content="No user input found for search.", name="error", metadata={"message_id": None})]
    recent_chat = state.get_recent_history_str()

    # Generate search query using LLM
    template = Template(config.loaded_prompts["web_search_prompt"])
    formatted_prompt = template.render(
        user_query=user_query.content, recent_chat_history=recent_chat
    )

    llm = ChatOpenAI(
        base_url=config.openai["base_url"],
        temperature=0.2,
        max_tokens=50,
        streaming=False,
        verbose=True,
        timeout=config.llm.timeout,
    )

    try:
        response = await llm.ainvoke([("system", formatted_prompt)])
        search_query = response.content.strip()

        # URL encode the search query
        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://serpapi.com/search.json?q={encoded_query}&api_key={SERPAPI_KEY}"
        resp = requests.get(url)
        data = resp.json()

        # Format results
        results = data.get("organic_results", [{"snippet": "No results"}])
        summary = "\n\n".join(
            [f"{i+1}. {item['snippet']}" for i, item in enumerate(results[:3])]
        )

        # Send Chainlit message
        cl_msg = cl.Message(
            content=f'**Search Results for "{search_query}":**\n\n{summary}',
            parent_id=None,
        )
        await cl_msg.send()

        return [
            AIMessage(
                content=f"Search results for '{search_query}':\n{summary}",
                name="web_search",
                metadata={"message_id": cl_msg.id},
            )
        ]
    except Exception as e:
        cl_logger.error(f"Search failed: {str(e)}")
        return [AIMessage(content=f"Search failed: {str(e)}", name="error", metadata={"message_id": None})]


@task
async def web_search(state: ChatState) -> list[BaseMessage]:
    return await _web_search(state)


web_search_agent = web_search
