import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.oracle_workflow import oracle_workflow
from src.models import ChatState


@pytest.fixture
def mock_chat_state():
    return ChatState(messages=[], thread_id="test-thread-id")


@pytest.mark.asyncio
async def test_oracle_workflow_memory_updates(mock_chat_state):
    dummy_ai_msg1 = AIMessage(
        content="Tool1 done", name="tool1", metadata={"message_id": "m1"}
    )
    dummy_knowledge_msg = AIMessage(
        content="Character info", name="character", metadata={"message_id": "k1"}
    )
    dummy_gm_msg = AIMessage(
        content="Story continues", name="Game Master", metadata={"message_id": "gm1"}
    )

    with patch(
        "src.oracle_workflow.persona_workflows",
        {
            "default": AsyncMock(
                return_value=[dummy_gm_msg]
            ),  # fallback persona workflow
        },
    ), patch(
        "src.oracle_workflow.append_log"
    ), patch(
        "src.oracle_workflow.get_persona_daily_dir"
    ), patch(
        "src.oracle_workflow.save_text_file"
    ), patch(
        "src.oracle_workflow.persona_classifier_agent",
        AsyncMock(return_value={"persona": "default"}),
    ), patch(
        "src.oracle_workflow.cl.user_session.get", new_callable=MagicMock
    ) as mock_user_session_get:

        vector_store = AsyncMock()
        mock_user_session_get.return_value = {}

        initial_state = mock_chat_state
        initial_state.messages.append(HumanMessage(content="Hi"))

        result_state = await oracle_workflow.ainvoke(
            {"messages": initial_state.messages, "previous": initial_state}
        )

        # Defensive: print messages for debugging
        # print([m.name for m in result_state.messages])

        assert isinstance(result_state, ChatState)
        assert any(
            m.content == "Story continues" for m in result_state.messages
        )


@pytest.mark.asyncio
async def test_oracle_workflow_dispatches_to_persona(monkeypatch):
    from src.oracle_workflow import oracle_workflow

    called = {}

    async def fake_secretary(inputs, state, **kwargs):
        called["secretary"] = True
        return []

    import src.persona_workflows as pw
    monkeypatch.setitem(pw.persona_workflows, "secretary", fake_secretary)

    state = ChatState(messages=[], thread_id="t", current_persona="secretary")
    await oracle_workflow.ainvoke({"messages": [], "previous": state})

    assert called.get("secretary")


@pytest.mark.asyncio
async def test_oracle_workflow_classifier_switch(monkeypatch):
    from src.oracle_workflow import oracle_workflow

    async def fake_classifier(state, **kwargs):
        return {"persona": "therapist", "reason": "User mentioned feelings"}

    async def fake_therapist(inputs, state, **kwargs):
        state.current_persona = "therapist"
        state.messages.append(
            AIMessage(content="Therapist response", name="therapist", metadata={})
        )
        return state.messages[-1:]

    import src.oracle_workflow as owf
    monkeypatch.setattr(owf, "persona_classifier_agent", fake_classifier)
    monkeypatch.setitem(
        owf.persona_workflows,
        "therapist",
        fake_therapist,
    )

    state = ChatState(messages=[], thread_id="t", current_persona=None)
    result_state = await oracle_workflow.ainvoke({"messages": [], "previous": state})

    assert result_state.current_persona == "therapist"
    assert any(
        m.content == "Therapist response" for m in result_state.messages
    )


@pytest.mark.asyncio
async def test_oracle_workflow_error_handling(monkeypatch):
    from src.oracle_workflow import oracle_workflow

    async def broken_workflow(inputs, state, **kwargs):
        raise RuntimeError("fail")

    import src.oracle_workflow as owf
    monkeypatch.setitem(owf.persona_workflows, "default", broken_workflow)

    state = ChatState(messages=[], thread_id="t", current_persona="default")
    result_state = await oracle_workflow.ainvoke({"messages": [], "previous": state})

    # Should append an error message
    assert any(
        m.name == "error" or "error" in m.content.lower() for m in result_state.messages
    )


# New test: all persona workflows run without error
import src.persona_workflows as pw

@pytest.mark.asyncio
@pytest.mark.parametrize("persona_key", list(pw.persona_workflows.keys()))
async def test_all_persona_workflows_run(persona_key):
    workflow = pw.persona_workflows[persona_key]
    state = ChatState(
        messages=[HumanMessage(content="Hello", name="Player")],
        thread_id="test",
        current_persona=persona_key,
    )
    result = await workflow({}, state)
    assert isinstance(result, list)


# New test: multi-hop tool call simulation
@pytest.mark.asyncio
async def test_oracle_workflow_multi_hop(monkeypatch):
    from src.oracle_workflow import oracle_workflow

    # Simulate a persona workflow that returns multiple messages
    async def fake_storyteller(inputs, state, **kwargs):
        state.messages.append(
            AIMessage(content="Lore info", name="lore", metadata={})
        )
        state.messages.append(
            AIMessage(content="Story continues", name="Game Master", metadata={})
        )
        return state.messages[-2:]

    import src.oracle_workflow as owf
    monkeypatch.setitem(owf.persona_workflows, "storyteller_gm", fake_storyteller)

    state = ChatState(
        messages=[HumanMessage(content="Tell me a story", name="Player")],
        thread_id="t",
        current_persona="storyteller_gm",
    )
    result_state = await oracle_workflow.ainvoke({"messages": state.messages, "previous": state})

    contents = [m.content for m in result_state.messages]
    assert "Lore info" in contents
    assert "Story continues" in contents
