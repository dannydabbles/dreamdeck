from src.event_handlers import on_chat_start
import chainlit as cl
from src import commands  # Ensure commands module is imported

import logging

# Register all event handlers and commands
from src.event_handlers import *
from src.commands import *

logging.getLogger("chainlit").info(
    "Dreamdeck app loaded with event handlers, commands, and Chainlit UI actions."
)
