from .agents.dice_agent import dice_roll_agent  # Now valid since dice_roll_agent exists
from .agents.web_search_agent import web_search_agent
from .agents.director_agent import director_agent
from .agents.writer_agent import writer_agent
from .agents.storyboard_editor_agent import storyboard_editor_agent
from .agents.knowledge_agent import knowledge_agent

# Agent dispatch map for Oracle workflow
agents_map = {
    "roll": dice_roll_agent,
    "dice_roll": dice_roll_agent,
    "web_search": web_search_agent,
    "director": director_agent,
    "writer": writer_agent,
    "storyboard_editor": storyboard_editor_agent,
    "continue_story": writer_agent,  # Map continue_story to writer_agent
    "knowledge": knowledge_agent,    # Map knowledge to knowledge_agent
    # Add more mappings as needed
}
