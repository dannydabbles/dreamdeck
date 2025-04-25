import chainlit as cl

from src.models import ChatState
from src.supervisor import supervisor

# The main app is now the supervisor agent (langgraph-supervisor based)
app = supervisor
app_without_checkpoint = supervisor


# Legacy alias for backward compatibility with old tests (calls supervisor)
async def _chat_workflow(*args, **kwargs):
    """
    Legacy alias for backward compatibility with old tests.
    Accepts (inputs_dict, state, config) or similar signatures.
    """
    # Accept both (inputs_dict, state, ...) and (state, ...)
    if args and isinstance(args[0], dict) and "state" in args[0]:
        state = args[0]["state"]
    elif args and isinstance(args[0], ChatState):
        state = args[0]
    elif "state" in kwargs:
        state = kwargs["state"]
    else:
        raise ValueError("No ChatState found in arguments to _chat_workflow")
    return await supervisor(state, **kwargs)


# Dummy stubs for test monkeypatching
agents_map = {}
