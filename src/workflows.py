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
    # Fix: avoid passing ChatState as config positional arg
    if len(args) >= 2 and isinstance(args[1], ChatState):
        # Remove ChatState positional arg from args, pass as 'state' kwarg instead
        args = (args[0],)
        kwargs['state'] = args[1]
    # Fix: avoid passing ChatState as config kwarg
    if 'config' in kwargs and isinstance(kwargs['config'], ChatState):
        kwargs['config'] = None
    return await oracle_workflow_runnable.ainvoke(*args, **kwargs)


# Dummy stubs for test monkeypatching
agents_map = {}
