"""
Exports all agent entrypoints for use in commands and workflows.
"""

from .web_search_agent import web_search_agent
from .dice_agent import dice_agent
from .writer_agent import writer_agent
from .director_agent import director_agent
from .todo_agent import todo_agent
from .knowledge_agent import knowledge_agent
from .persona_classifier_agent import persona_classifier_agent
from .report_agent import report_agent

__all__ = [
    "web_search_agent",
    "dice_agent",
    "writer_agent",
    "director_agent",
    "todo_agent",
    "knowledge_agent",
    "persona_classifier_agent",
    "report_agent",
]

agents_map = {
    "roll": dice_agent,
    "search": web_search_agent,
    "todo": todo_agent,
    # "knowledge" handled explicitly in workflow
    "write": writer_agent,
    "continue_story": writer_agent,
    "report": report_agent,
}
