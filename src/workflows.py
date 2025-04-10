from src.oracle_workflow import oracle_workflow

app = oracle_workflow

app_without_checkpoint = app


async def _chat_workflow(*args, **kwargs):
    """Deprecated legacy workflow stub to satisfy old tests."""
    raise NotImplementedError(
        "The legacy _chat_workflow has been removed. Update tests to use oracle_workflow instead."
    )
