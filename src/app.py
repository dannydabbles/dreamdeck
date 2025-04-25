import logging

import chainlit as cl

from src import commands  # Ensure commands module is imported
from src.commands import *

# Register all event handlers and commands
from src.event_handlers import *
from src.event_handlers import on_chat_start

logging.getLogger("chainlit").info(
    "Dreamdeck app loaded with event handlers, commands, and Chainlit UI actions."
)
