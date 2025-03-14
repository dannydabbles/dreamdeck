from src.config import config
import os
import requests
import logging
from jinja2 import Template
from uuid import uuid4  # Import uuid4
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # Use LangChain's standard messages
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..config import SERPAPI_KEY, WEB_SEARCH_ENABLED
from ..models import ChatState

import chainlit as cl

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _web_search(state: ChatState) -> list[BaseMessage]:
    """Execute web search using natural language processing"""
    if not WEB_SEARCH_ENABLED:
        return [AIMessage(content="Web search is disabled.", name="error")]
    
    # Get user input and context
    user_query = next((m for m in reversed(state.messages) if isinstance(m, HumanMessage)), "")
    recent_chat = state.get_recent_history_str()
    
    # Generate search query using LLM
    template = Template(config.prompts['web_search_prompt'])
    formatted_prompt = template.render(
        user_query=user_query.content,
        recent_chat_history=recent_chat
    )
    
    llm = ChatOpenAI(
        base_url=config.openai["base_url"],
        temperature=0.2,
        max_tokens=50,
        streaming=False,
        verbose=True,
        timeout=config.llm.timeout
    )
    
    try:
        response = llm.invoke([('system', formatted_prompt)])  # Add 'await'
        search_query = response.content.strip()
            
        # Proceed with search execution
        url = f"https://serpapi.com/search.json?q={search_query}&api_key={SERPAPI_KEY}"
        resp = requests.get(url)
        data = resp.json()
            
        # Format results
        results = data.get("organic_results", [{"snippet": "No results"}])
        summary = "\n\n".join([f"{i+1}. {item['snippet']}" for i,item in enumerate(results[:3])])
            
        # Send Chainlit message
        cl_msg = cl.Message(
            content=f"**Search Results for \"{search_query}\":**\n\n{summary}",
            parent_id=None
        )
        await cl_msg.send()
            
        return [
            AIMessage(
                content=f"Search results for '{search_query}':\n{summary}",
                name="web_search"
            )
        ]
    except Exception as e:
        cl_logger.error(f"Search failed: {str(e)}")
        return [AIMessage(content=f"Search failed: {str(e)}", name="error")]

@task
async def web_search(state: ChatState) -> list[BaseMessage]:
    return await _web_search(state)

web_search_agent = web_search
