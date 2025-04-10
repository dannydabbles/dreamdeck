from src.oracle_workflow import oracle_workflow

app = oracle_workflow

app_without_checkpoint = app


async def _chat_workflow(*args, **kwargs):
    """Legacy alias kept for backward compatibility with old tests."""
    return await oracle_workflow(*args, **kwargs)
