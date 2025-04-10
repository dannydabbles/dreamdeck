from src.oracle_workflow import oracle_workflow

app = oracle_workflow

app_without_checkpoint = app


async def _chat_workflow(*args, **kwargs):
    """Legacy alias kept for backward compatibility with old tests."""
    return await oracle_workflow(*args, **kwargs)


# Dummy stubs for test monkeypatching
async def director_agent(*args, **kwargs):
    raise NotImplementedError("director_agent stub")


async def knowledge_agent(*args, **kwargs):
    raise NotImplementedError("knowledge_agent stub")


async def writer_agent(*args, **kwargs):
    raise NotImplementedError("writer_agent stub")


async def storyboard_editor_agent(*args, **kwargs):
    raise NotImplementedError("storyboard_editor_agent stub")


agents_map = {}
