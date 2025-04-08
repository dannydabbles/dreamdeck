"""
Exports all agent entrypoints for use in commands and workflows.
"""

from .web_search_agent import web_search_agent
from .dice_agent import dice_agent
from .writer_agent import writer_agent
from .orchestrator_agent import orchestrator_agent
from .todo_agent import todo_agent
from .character_agent import character_agent
from .lore_agent import lore_agent
from .puzzle_agent import puzzle_agent

__all__ = [
    "web_search_agent",
    "dice_agent",
    "writer_agent",
    "orchestrator_agent",
    "todo_agent",
    "character_agent",
    "lore_agent",
    "puzzle_agent",
]

agents_map = {
    "roll": dice_agent,
    "search": web_search_agent,
    "todo": todo_agent,
    "character": character_agent,
    "lore": lore_agent,
    "puzzle": puzzle_agent,
    "write": writer_agent,
    "continue_story": writer_agent,
}
