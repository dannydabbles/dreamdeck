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
    """Generates the final response for the Secretary persona."""
    try:
        # The Oracle loop handles calling the 'todo' agent if needed.
        # This workflow now only generates the final narrative/summary response.
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

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
    """Generates the final response for the Lorekeeper persona."""
    try:
        # The Oracle loop handles calling the 'knowledge' agent if needed.
        # This workflow now only generates the final narrative/summary response.
        story_messages = await _generate_story(state)

        # Patch: ensure all AIMessage outputs have consistent metadata
        if isinstance(story_messages, list):
            for msg in story_messages:
                if isinstance(msg, AIMessage):
                    if msg.metadata is None:
                        msg.metadata = {}
                    msg.metadata["type"] = "ai"
                    msg.metadata["persona"] = state.current_persona

        return story_messages
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
    # PATCH: allow "continue_story" to be routed to storyteller_workflow for test compatibility
    "continue_story": storyteller_workflow,
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
