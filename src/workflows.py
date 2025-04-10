from src.agents.director_agent import director_agent
from src.agents.writer_agent import writer_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.oracle_workflow import oracle_workflow_runnable, chat_workflow

import chainlit as cl

app = chat_workflow
app_without_checkpoint = chat_workflow


async def _chat_workflow(*args, **kwargs):
    """Legacy alias kept for backward compatibility with old tests."""
    return await oracle_workflow_runnable.ainvoke(*args, **kwargs)


# Dummy stubs for test monkeypatching
agents_map = {}
