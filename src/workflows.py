from src.agents.writer_agent import writer_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.oracle_workflow import oracle_workflow_runnable, chat_workflow
from src.models import ChatState

import chainlit as cl

app = chat_workflow
app_without_checkpoint = chat_workflow


async def _chat_workflow(*args, **kwargs):
    """Legacy alias kept for backward compatibility with old tests."""
    inputs_dict = {}
    state_arg = None
    config = kwargs.pop('config', None)  # Extract config early

    # Handle positional arguments first
    if len(args) >= 1 and isinstance(args[0], dict):
        inputs_dict = args[0]
        # If state is already in the dict, use it
        if 'state' in inputs_dict and isinstance(inputs_dict['state'], ChatState):
            state_arg = inputs_dict['state']
    if len(args) >= 2 and isinstance(args[1], ChatState):
        # If state is passed as second positional arg, prioritize it
        state_arg = args[1]

    # Handle keyword arguments (state might be passed via kwargs)
    if 'state' in kwargs and isinstance(kwargs['state'], ChatState):
        state_arg = kwargs.pop('state')  # Remove state from kwargs

    # Ensure state is correctly placed in the inputs dictionary
    if state_arg:
        inputs_dict['state'] = state_arg
        # Ensure 'previous' key also points to the state if needed by the workflow logic
        inputs_dict.setdefault('previous', state_arg)
        # Ensure 'messages' are present if state is available
        inputs_dict.setdefault('messages', state_arg.messages)
    elif 'previous' in inputs_dict and isinstance(inputs_dict['previous'], ChatState):
        # If only 'previous' is present, use it as state
        state_arg = inputs_dict['previous']
        inputs_dict['state'] = state_arg
        inputs_dict.setdefault('messages', state_arg.messages)

    # Pass the consolidated inputs dictionary and the extracted config
    # Pass remaining kwargs as well, though they shouldn't contain 'state' anymore
    return await oracle_workflow_runnable.ainvoke(inputs_dict, config=config, **kwargs)


# Dummy stubs for test monkeypatching
agents_map = {}
