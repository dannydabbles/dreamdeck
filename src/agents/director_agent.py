from langgraph.func import entrypoint, task

@task
def director_decision(user_input: str) -> str:
    """Generate a high-level narrative directive."""
    return f"Plot: {user_input} leads to a new challenge."

@entrypoint
def director_agent(user_input: str, *, previous: dict = None) -> dict:
    state = previous or {}
    directive = director_decision(user_input).result()
    gm_response = gm_agent(f"Directive: {directive}").result()
    state["directive"] = directive
    state["gm_response"] = gm_response
    return entrypoint.final(value=state, save=state)
