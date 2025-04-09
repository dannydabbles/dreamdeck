from src.config import config, OPENAI_SETTINGS, LLM_TIMEOUT
import logging
from jinja2 import Template
from langgraph.func import task
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.models import ChatState

import chainlit as cl

cl_logger = logging.getLogger("chainlit")


@cl.step(name="Knowledge Agent", type="tool")
async def _knowledge(state: ChatState, knowledge_type: str) -> list[BaseMessage]:
    """Generates knowledge content (character, lore, puzzle) based on type."""
    try:
        persona = getattr(state, "current_persona", "Default")
        prompt_key = f"{knowledge_type}_prompt"
        try:
            persona_configs = getattr(config.agents, "knowledge_agent", {}).get("personas", {})
            if isinstance(persona_configs, dict):
                persona_entry = persona_configs.get(persona)
                if persona_entry and isinstance(persona_entry, dict):
                    prompt_key = persona_entry.get("prompt_key", prompt_key)
        except Exception:
            pass  # fallback to default

        default_prompt_text = f"Generate some {knowledge_type} information."
        prompt_template_str = config.loaded_prompts.get(prompt_key, default_prompt_text)

        template = Template(prompt_template_str)
        formatted_prompt = template.render(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
            user_preferences=state.user_preferences,
        )

        # Get user settings dynamically based on knowledge_type, fallback to defaults
        user_settings = cl.user_session.get("chat_settings", {})
        default_temp = 0.7
        default_max_tokens = 500

        final_temp = user_settings.get(f"{knowledge_type}_temp", default_temp)
        final_endpoint = user_settings.get(f"{knowledge_type}_endpoint") or OPENAI_SETTINGS.get("base_url")
        final_max_tokens = user_settings.get(f"{knowledge_type}_max_tokens", default_max_tokens)

        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=False,
            verbose=True,
            timeout=LLM_TIMEOUT,
        )

        response = await llm.ainvoke([("system", formatted_prompt)])
        cl_msg = cl.Message(content=response.content.strip())
        await cl_msg.send()
        return [
            AIMessage(
                content=response.content.strip(),
                name=knowledge_type,
                metadata={"message_id": cl_msg.id},
            )
        ]
    except Exception as e:
        cl_logger.error(f"Knowledge agent ({knowledge_type}) failed: {e}")
        return [AIMessage(content=f"{knowledge_type.capitalize()} generation failed.", name="error", metadata={"message_id": None})]


@task
async def knowledge_agent(state: ChatState, knowledge_type: str, **kwargs) -> list[BaseMessage]:
    return await _knowledge(state, knowledge_type)
