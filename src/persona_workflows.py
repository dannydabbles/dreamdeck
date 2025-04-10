from src.models import ChatState
from langchain_core.messages import BaseMessage
from src.agents.writer_agent import writer_agent
from src.agents.todo_agent import manage_todo
from src.agents.knowledge_agent import knowledge_agent
from src.storage import (
    get_persona_daily_dir,
    get_shared_daily_dir,
    save_text_file,
    load_text_file,
)
import datetime
import os

async def storyteller_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    # Optionally update state with inputs here
    # For now, just call the writer agent (which already uses persona-aware prompts)
    return await writer_agent(state)

async def therapist_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    # Could add therapist-specific pre-processing here
    return await writer_agent(state)

async def secretary_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    # Load today's TODO file for secretary persona
    today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
    persona_dir = get_persona_daily_dir(state.current_persona, today)
    todo_path = persona_dir / "todo.md"
    if todo_path.exists():
        todo_content = load_text_file(todo_path)
        # Optionally, attach to state metadata or memories
        state.metadata["todo"] = todo_content

    # Run TODO update first
    await manage_todo(state)

    # Save updated TODO back to file
    # The todo agent writes to file, so this is redundant, but safe
    if "todo" in state.metadata:
        save_text_file(todo_path, state.metadata["todo"])

    # Then generate secretary-style response
    return await writer_agent(state)

async def coder_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    return await writer_agent(state)

async def friend_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    return await writer_agent(state)

async def lorekeeper_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    # Default to lore knowledge type
    return await knowledge_agent(state, knowledge_type="lore")

async def dungeon_master_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    return await writer_agent(state)

async def default_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
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
