import pytest
from src.persona_workflows import persona_workflows
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persona_key",
    [
        "storyteller_gm",
        "therapist",
        "secretary",
        "coder",
        "friend",
        "lorekeeper",
        "dungeon_master",
        "default",
    ],
)
async def test_persona_workflow_runs(persona_key):
    workflow = persona_workflows[persona_key]
    state = ChatState(
        messages=[HumanMessage(content="Hello", name="Player")],
        thread_id="test",
        current_persona=persona_key,
    )
    result = await workflow({}, state)
    assert isinstance(result, list)
