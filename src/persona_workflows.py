from src.models import ChatState
from langchain_core.messages import BaseMessage
from src.agents.writer_agent import writer_agent
from src.agents.todo_agent import manage_todo
from src.agents.knowledge_agent import knowledge_agent

async def storyteller_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    # Optionally update state with inputs here
    # For now, just call the writer agent (which already uses persona-aware prompts)
    return await writer_agent(state)

async def therapist_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    # Could add therapist-specific pre-processing here
    return await writer_agent(state)

async def secretary_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    # Optionally, run TODO update first
    await manage_todo(state)
    # Then generate secretary-style response
    return await writer_agent(state)

async def coder_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    return await writer_agent(state)

async def friend_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    return await writer_agent(state)

async def lorekeeper_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    # Default to lore knowledge type
    return await knowledge_agent(state, knowledge_type="lore")

async def dungeon_master_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    return await writer_agent(state)

async def default_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
    return await writer_agent(state)

persona_workflows = {
    "storyteller_gm": storyteller_workflow,
    "therapist": therapist_workflow,
    "secretary": secretary_workflow,
    "coder": coder_workflow,
    "friend": friend_workflow,
    "lorekeeper": lorekeeper_workflow,
    "dungeon_master": dungeon_master_workflow,
    "default": default_workflow,
}

__all__ = [
    "storyteller_workflow",
    "therapist_workflow",
    "secretary_workflow",
    "coder_workflow",
    "friend_workflow",
    "lorekeeper_workflow",
    "dungeon_master_workflow",
    "default_workflow",
    "persona_workflows",
]
