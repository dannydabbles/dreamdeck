"""
Exports all agent entrypoints for use in commands and workflows.
"""

from .web_search_agent import _web_search
from .dice_agent import _dice_roll
from .writer_agent import writer_agent
from .todo_agent import _manage_todo
from .knowledge_agent import _knowledge as knowledge_agent_func
from .report_agent import _generate_report
# Phase 1: Import persona workflows to add them to the map
from src.persona_workflows import (
    storyteller_workflow,
    therapist_workflow,
    secretary_workflow,
    coder_workflow,
    friend_workflow,
    lorekeeper_workflow,
    dungeon_master_workflow,
    default_workflow,
    persona_workflows, # Also import the dict for convenience
)


__all__ = [
    "web_search_agent",
    "dice_roll",
    "writer_agent",
    "manage_todo",
    "knowledge_agent",
    "report_agent",
]

# Expose agents_map as a module attribute for patching
from .dice_agent import _dice_roll

# Always use the undecorated (raw) functions for tool agents in agents_map
import os
agents_map = {
    "roll": _dice_roll,
    "dice_roll": _dice_roll,
    "search": _web_search,
    "web_search": _web_search,
    "todo": _manage_todo,
    "manage_todo": _manage_todo,
    "write": writer_agent._generate_story,
    # PATCH: Map continue_story to the correct workflow function, not the tool
    # NOTE: Must be called as (inputs, state, config) for persona workflows
    "continue_story": storyteller_workflow,
    "report": _generate_report,
    "knowledge": knowledge_agent_func,  # Patch: expose as function, not partial
    # Phase 1: Add persona agents to the map using their keys
    "storyteller_gm": storyteller_workflow,
    "therapist": therapist_workflow,
    "secretary": secretary_workflow,
    "coder": coder_workflow,
    "friend": friend_workflow,
    "lorekeeper": lorekeeper_workflow,
    "dungeon_master": dungeon_master_workflow,
    "default": default_workflow,
}
import sys as _sys

_sys.modules[__name__].agents_map = agents_map
