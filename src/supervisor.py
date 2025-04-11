"""
Supervisor agent for Dreamdeck, orchestrating persona agents and tools
using the langgraph-supervisor pattern.
"""

from src.models import ChatState
from src.agents.writer_agent import writer_agent
from src.agents.dice_agent import dice_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.todo_agent import todo_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.report_agent import report_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.persona_classifier_agent import persona_classifier_agent

from langchain_core.messages import HumanMessage, AIMessage
from langgraph.func import task

import logging

cl_logger = logging.getLogger("chainlit")

# Map tool/agent names to callables
TOOL_MAP = {
    "roll": dice_agent,
    "dice_roll": dice_agent,
    "search": web_search_agent,
    "web_search": web_search_agent,
    "todo": todo_agent,
    "knowledge": knowledge_agent,
    "report": report_agent,
    "storyboard_editor": storyboard_editor_agent,
    "storyboard": storyboard_editor_agent,
    "persona_classifier": persona_classifier_agent,
}

# Map persona names to persona agents (case-insensitive)
PERSONA_MAP = writer_agent.persona_agent_registry

def _normalize_persona(persona: str) -> str:
    if not persona:
        return "default"
    return persona.lower().replace(" ", "_")

@task
async def supervisor(state: ChatState, **kwargs):
    """
    Supervisor agent: routes user input to the correct tool or persona agent.
    """
    last_human = state.get_last_human_message()
    if not last_human:
        cl_logger.warning("Supervisor: No user message found in state.")
        return []

    user_input = last_human.content.strip().lower()

    # Tool routing: check for explicit tool commands
    if user_input.startswith("/roll") or "roll" in user_input:
        cl_logger.info("Supervisor: Routing to dice_agent.")
        return await dice_agent(state)
    if user_input.startswith("/search") or "search" in user_input:
        cl_logger.info("Supervisor: Routing to web_search_agent.")
        return await web_search_agent(state)
    if user_input.startswith("/todo") or "todo" in user_input:
        cl_logger.info("Supervisor: Routing to todo_agent.")
        return await todo_agent(state)
    if user_input.startswith("/report") or "report" in user_input:
        cl_logger.info("Supervisor: Routing to report_agent.")
        return await report_agent(state)
    if user_input.startswith("/storyboard") or "storyboard" in user_input:
        cl_logger.info("Supervisor: Routing to storyboard_editor_agent.")
        # Find last GM message id for storyboard
        gm_msg = next(
            (msg for msg in reversed(state.messages)
             if isinstance(msg, AIMessage) and msg.name and "game master" in msg.name.lower()
             and msg.metadata and "message_id" in msg.metadata),
            None,
        )
        if gm_msg:
            return await storyboard_editor_agent(state, gm_message_id=gm_msg.metadata["message_id"])
        else:
            cl_logger.warning("Supervisor: No GM message found for storyboard.")
            return []

    # Persona classifier: suggest persona switch if needed
    # (This is handled in event_handlers, but supervisor could also do it if desired.)

    # Default: route to current persona agent
    persona = getattr(state, "current_persona", "default")
    persona_key = _normalize_persona(persona)
    agent = PERSONA_MAP.get(persona_key, writer_agent)
    cl_logger.info(f"Supervisor: Routing to persona agent '{persona_key}'.")
    return await agent(state)
