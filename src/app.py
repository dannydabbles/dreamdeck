from src.event_handlers import on_chat_start
import chainlit as cl
from src import commands  # Ensure commands module is imported

import logging

# Set a default persona at chat start if none exists
@cl.on_chat_start
async def _set_default_persona():
    if not cl.user_session.get("current_persona"):
        cl.user_session.set("current_persona", "Storyteller GM")
        await cl.Message(
            content="Starting chat with the **Storyteller GM** persona."
        ).send()
    else:
        # Show current persona on chat start
        persona = cl.user_session.get("current_persona")
        await cl.Message(content=f"Continuing with persona **{persona}**.").send()

# Register all event handlers and commands
from src.event_handlers import *
from src.commands import *

logging.getLogger("chainlit").info(
    "Dreamdeck app loaded with default persona and event handlers."
)
