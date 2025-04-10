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
    dummy_gm_msg = AIMessage(
        content="Story continues", name="Game Master", metadata={"message_id": "gm1"}
    )

    # Patch persona_workflows dict
    with patch(
        "src.oracle_workflow.persona_workflows",
        {"default": AsyncMock(return_value=[dummy_gm_msg])},
    ), patch(
        "src.oracle_workflow.persona_classifier_agent",
        AsyncMock(return_value={"persona": "default"}),
    ), patch(
        "src.oracle_workflow.cl.user_session.get", new_callable=MagicMock
    ) as mock_user_session_get:

        mock_user_session_get.return_value = {}

        # Patch append_log etc. which are imported inside oracle_workflow()
        import src.oracle_workflow as owf
        owf.append_log = lambda *a, **kw: None
        owf.get_persona_daily_dir = lambda *a, **kw: MagicMock()
        owf.save_text_file = lambda *a, **kw: None

        initial_state = mock_chat_state
        initial_state.messages.append(HumanMessage(content="Hi"))

        result_state = await oracle_workflow.ainvoke(
            {"messages": initial_state.messages, "previous": initial_state},
            initial_state,
        )

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
    await oracle_workflow.ainvoke(
        {"messages": [], "previous": state},
        state,
    )

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

    # current_persona cannot be None (pydantic validation), so set to empty string
    state = ChatState(messages=[], thread_id="t", current_persona="")
    result_state = await oracle_workflow.ainvoke(
        {"messages": [], "previous": state},
        state,
    )

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
    result_state = await oracle_workflow.ainvoke(
        {"messages": [], "previous": state},
        state,
    )

    # Should append an error message
    assert any(
        m.name == "error" or "error" in m.content.lower() for m in result_state.messages
    )


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


@pytest.mark.asyncio
async def test_oracle_workflow_multi_hop(monkeypatch):
    from src.oracle_workflow import oracle_workflow

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
    result_state = await oracle_workflow.ainvoke(
        {"messages": state.messages, "previous": state},
        state,
    )

    contents = [m.content for m in result_state.messages]
    assert "Lore info" in contents
    assert "Story continues" in contents
