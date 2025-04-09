from src.event_handlers import on_chat_start
import chainlit as cl
from src import commands  # Ensure commands module is imported

@cl.profile(name="Storyteller GM")
async def storyteller_gm():
    """Profile for the standard Game Master persona."""
    cl.user_session.set("current_persona", "Storyteller GM")
    await cl.Message(content="Starting chat with the **Storyteller GM** persona.").send()


@cl.profile(name="Default")
async def default_profile():
    """Default profile if none is selected."""
    cl.user_session.set("current_persona", "Default")
    await cl.Message(content="Starting chat with the **Default** persona.").send()
