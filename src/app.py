from src.event_handlers import on_chat_start
import chainlit as cl
from src import commands  # Ensure commands module is imported

import logging

# Set a default persona at chat start if none exists
@cl.on_chat_start
async def _set_default_persona():
    if not cl.user_session.get("current_persona"):
        cl.user_session.set("current_persona", "Storyteller GM")
        await cl.Message(content="Starting chat with the **Storyteller GM** persona.").send()

from src.event_handlers import *  # Register all event handlers
from src.commands import *        # Register slash commands

logging.getLogger("chainlit").info("Dreamdeck app loaded with default persona and event handlers.")
