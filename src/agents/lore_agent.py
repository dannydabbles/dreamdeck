from src.config import config
import logging
from jinja2 import Template
from langgraph.func import task
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.models import ChatState

import chainlit as cl

cl_logger = logging.getLogger("chainlit")


async def _lore(state: ChatState) -> list[BaseMessage]:
    try:
        template = Template(config.loaded_prompts.get("lore_prompt", "Generate some world lore."))
        formatted_prompt = template.render(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
        )

        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("lore_temp", 0.7)
        final_endpoint = user_settings.get("lore_endpoint") or config.OPENAI_SETTINGS.get("base_url")
        final_max_tokens = user_settings.get("lore_max_tokens", 500)

        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=False,
            verbose=True,
            timeout=config.LLM_TIMEOUT,
        )

        response = await llm.ainvoke([("system", formatted_prompt)])
        return [AIMessage(content=response.content.strip(), name="lore")]
    except Exception as e:
        cl_logger.error(f"Lore agent failed: {e}")
        return [AIMessage(content="Lore generation failed.", name="error")]


@task
async def lore_agent(state: ChatState, **kwargs) -> list[BaseMessage]:
    return await _lore(state)
