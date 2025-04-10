from src.models import ChatState
from langchain_core.messages import BaseMessage, AIMessage
from src.agents.writer_agent import writer_agent
from src.agents.writer_agent import _generate_story
from src.agents.todo_agent import _manage_todo
from src.agents.knowledge_agent import knowledge_agent
from src.storage import (
    get_persona_daily_dir,
    get_shared_daily_dir,
    save_text_file,
    load_text_file,
    append_log,
)
import datetime
import os
import logging

cl_logger = logging.getLogger("chainlit")


async def storyteller_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata for downstream processing and tests
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return story_messages
    except Exception as e:
        cl_logger.error(f"storyteller_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Story generation failed.", name="error", metadata={"message_id": None})]


async def therapist_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata for downstream processing and tests
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return story_messages
    except Exception as e:
        cl_logger.error(f"therapist_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Therapist response failed.", name="error", metadata={"message_id": None})]


async def secretary_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        persona_dir = get_persona_daily_dir(state.current_persona, today)
        todo_path = persona_dir / "todo.md"
        if todo_path.exists():
            todo_content = load_text_file(todo_path)
            state.metadata["todo"] = todo_content

        todo_messages = await _manage_todo(state)

        # Patch: ensure all AIMessage outputs from _manage_todo have correct metadata
        if isinstance(todo_messages, list):
            for msg in todo_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        # Add todo_messages to state.messages
        if todo_messages:
            state.messages.extend(todo_messages)

        if "todo" in state.metadata:
            save_text_file(todo_path, state.metadata["todo"])

        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs from _generate_story have correct metadata
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        # Add story_messages to state.messages
        if story_messages:
            state.messages.extend(story_messages)

        # Return combined list for downstream
        return story_messages

    except Exception as e:
        cl_logger.error(f"secretary_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Secretary response failed.", name="error", metadata={"message_id": None})]


async def coder_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata for downstream processing and tests
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return story_messages
    except Exception as e:
        cl_logger.error(f"coder_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Coder response failed.", name="error", metadata={"message_id": None})]


async def friend_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata for downstream processing and tests
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return story_messages
    except Exception as e:
        cl_logger.error(f"friend_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Friend response failed.", name="error", metadata={"message_id": None})]


async def lorekeeper_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        knowledge_messages = await knowledge_agent(state, knowledge_type="lore")

        # Patch: ensure all AIMessage outputs have consistent metadata for downstream processing and tests
        if isinstance(knowledge_messages, list):
            for msg in knowledge_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return knowledge_messages
    except Exception as e:
        cl_logger.error(f"lorekeeper_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Lorekeeper response failed.", name="error", metadata={"message_id": None})]


async def dungeon_master_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata for downstream processing and tests
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return story_messages
    except Exception as e:
        cl_logger.error(f"dungeon_master_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Dungeon Master response failed.", name="error", metadata={"message_id": None})]


async def default_workflow(inputs: dict, state: ChatState, *, config=None) -> list[BaseMessage]:
    try:
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata for downstream processing and tests
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return story_messages
    except Exception as e:
        cl_logger.error(f"default_workflow failed: {e}", exc_info=True)
        append_log(state.current_persona, f"Error: {str(e)}")
        return [AIMessage(content="Default response failed.", name="error", metadata={"message_id": None})]


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
