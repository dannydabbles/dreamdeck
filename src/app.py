from src.event_handlers import on_chat_start
import chainlit as cl
from src import commands  # Ensure slash commands are registered

cl.register_slash_command(
    name="roll",
    description="Roll dice (e.g., /roll 2d6 or /roll check perception)",
    run=commands.command_roll,
)

cl.register_slash_command(
    name="search",
    description="Perform a web search (e.g., /search history of dragons)",
    run=commands.command_search,
)

cl.register_slash_command(
    name="todo",
    description="Add a TODO item (e.g., /todo Remember to buy milk)",
    run=commands.command_todo,
)

cl.register_slash_command(
    name="write",
    description="Directly prompt the writer agent (e.g., /write The wizard casts a spell)",
    run=commands.command_write,
)

cl.register_slash_command(
    name="storyboard",
    description="Generate storyboard for the last scene",
    run=commands.command_storyboard,
)

cl.register_slash_command(
    name="help",
    description="List all available slash commands",
    run=commands.command_help,
)

cl.register_slash_command(
    name="reset",
    description="Reset the current story and start fresh",
    run=commands.command_reset,
)

cl.register_slash_command(
    name="save",
    description="Export the current story as a markdown file",
    run=commands.command_save,
)
