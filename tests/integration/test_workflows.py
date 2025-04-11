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
    # Patch Oracle agent instead of Director
    ), patch( # Patch oracle_agent within the oracle_workflow module
        "src.oracle_workflow.oracle_agent",
        AsyncMock(return_value="default"), # Oracle decides to call default persona workflow
    ), patch( # Patch agents_map within the oracle_workflow module
        "src.oracle_workflow.agents_map",
        {"default": AsyncMock(return_value=[dummy_gm_msg])} # Map 'default' action to the mocked workflow
    ), patch( # Patch cl.user_session.get within the oracle_workflow module
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
            initial_state, # Pass state object as second arg
            config={"configurable": {"thread_id": initial_state.thread_id}} # Pass config
        )

        assert isinstance(result_state, ChatState), f"Expected ChatState, got {type(result_state)}"
        assert any(
            m.content == "Story continues" for m in result_state.messages
        ), "Expected dummy GM message not found"


@pytest.mark.asyncio
async def test_oracle_workflow_dispatches_to_persona(monkeypatch):
    from src.oracle_workflow import oracle_workflow

    # Patch persona_workflows to dummy workflows to avoid real LLM calls
    import src.persona_workflows as pw

    called = {}

    async def fake_secretary(inputs, state, **kwargs):
        called["secretary"] = True
        return [
            __import__("langchain_core.messages", fromlist=["AIMessage"]).AIMessage(
                content="Dummy secretary response",
                name="secretary",
                metadata={"message_id": "dummy_id"},
            )
        ]

    # Patch *all* persona_workflows to dummy_workflow, except 'secretary' to fake_secretary
    async def dummy_workflow(inputs, state, **kwargs):
        from langchain_core.messages import AIMessage
        return [
            AIMessage(
                content=f"Dummy response for persona {state.current_persona}",
                name=state.current_persona,
                metadata={"message_id": "dummy_id"},
            )
        ]

    for key in pw.persona_workflows:
        monkeypatch.setitem(pw.persona_workflows, key, dummy_workflow)

    monkeypatch.setitem(pw.persona_workflows, "secretary", fake_secretary) # Keep this

    # Patch the classifier in the oracle_workflow module
    async def fake_classifier(state, **kwargs):
         return {"persona": "secretary", "reason": "test"}
    monkeypatch.setattr("src.oracle_workflow.persona_classifier_agent", fake_classifier)

    # Patch the oracle agent to call the secretary workflow
    async def fake_oracle(state, **kwargs):
         return "secretary" # Oracle decides to call the persona workflow
    monkeypatch.setattr("src.oracle_workflow.oracle_agent", fake_oracle)

    # Patch the agents_map within oracle_workflow to include the patched secretary
    import src.oracle_workflow as owf
    agents_map_patch = owf.agents_map.copy() # Start with real map from the module
    agents_map_patch["secretary"] = fake_secretary # Override secretary
    monkeypatch.setattr(owf, "agents_map", agents_map_patch)

    # Patch necessary cl functions within oracle_workflow
    monkeypatch.setattr(owf.cl, "user_session", MagicMock(get=lambda k,d=None: {}))
    monkeypatch.setattr(owf, "append_log", lambda *a, **kw: None)


    state = ChatState(messages=[], thread_id="t", current_persona="default") # Start as default
    # Run the workflow - it should classify, then oracle calls secretary
    # Pass state object as second argument
    final_state = await oracle_workflow(
        {"messages": [], "previous": state, "force_classify": True}, # Force classify
        state, # Pass state object
         config={"configurable": {"thread_id": "t"}}
    )

    assert isinstance(final_state, ChatState), f"Expected ChatState, got {type(final_state)}"
    assert called.get("secretary"), "fake_secretary workflow was not called"
    assert final_state.current_persona == "secretary", "Final state persona should be secretary"


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
    # Patch oracle_agent instead of director_agent
    monkeypatch.setattr(owf, "oracle_agent", AsyncMock(return_value="therapist")) # Oracle calls therapist workflow
    # Patch agents_map within oracle_workflow
    agents_map_patch = owf.agents_map.copy()
    agents_map_patch["therapist"] = fake_therapist # Map therapist action to the fake workflow
    monkeypatch.setattr(owf, "agents_map", agents_map_patch)

    # Patch necessary cl functions within oracle_workflow
    monkeypatch.setattr(owf.cl, "user_session", MagicMock(get=lambda k,d=None: {}))
    monkeypatch.setattr(owf, "append_log", lambda *a, **kw: None)

    # current_persona cannot be None (pydantic validation), so set to default
    state = ChatState(messages=[], thread_id="t", current_persona="default")
    # Pass state object as second argument, force classify
    result_state = await oracle_workflow(
        {"messages": [], "previous": state, "force_classify": True},
        state, # Pass state object
        config={"configurable": {"thread_id": "t"}}
    )

    assert isinstance(result_state, ChatState), f"Expected ChatState, got {type(result_state)}"
    assert result_state.current_persona == "therapist"
    assert any(
        m.content == "Therapist response" for m in result_state.messages
    ), "Therapist response message not found"


@pytest.mark.asyncio
async def test_oracle_workflow_error_handling(monkeypatch):
    from src.oracle_workflow import oracle_workflow

    async def broken_workflow(inputs, state, **kwargs):
        raise RuntimeError("fail")

    import src.oracle_workflow as owf
    monkeypatch.setitem(owf.persona_workflows, "default", broken_workflow)

    # Patch classifier and oracle
    monkeypatch.setattr("src.oracle_workflow.persona_classifier_agent", AsyncMock(return_value={"persona": "default"}))
    monkeypatch.setattr("src.oracle_workflow.oracle_agent", AsyncMock(return_value="default")) # Oracle calls the broken workflow

    # Patch agents_map in oracle_workflow module
    agents_map_patch = owf.agents_map.copy()
    agents_map_patch["default"] = broken_workflow
    monkeypatch.setattr(owf, "agents_map", agents_map_patch) # Patch map on the module

    # Patch necessary cl functions within oracle_workflow
    monkeypatch.setattr(owf.cl, "user_session", MagicMock(get=lambda k,d=None: {}))
    monkeypatch.setattr(owf, "append_log", lambda *a, **kw: None)


    state = ChatState(messages=[], thread_id="t", current_persona="default")
    # Pass state object as second argument
    result_state = await oracle_workflow(
        {"messages": [], "previous": state},
        state, # Pass state object
         config={"configurable": {"thread_id": "t"}}
    )

    assert isinstance(result_state, ChatState), f"Workflow did not return ChatState, got {type(result_state)}"
    # Should append an error message from the Oracle loop's error handling
    assert any(
        m.name == "error" for m in result_state.messages
    ), f"Error message not found in state messages: {[m.name for m in result_state.messages]}"


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
        # Return only the new messages added by this workflow
        return state.messages[-2:]

    import src.oracle_workflow as owf
    # Patch oracle_agent instead of director_agent
    # Simulate Oracle deciding to call the storyteller_gm workflow
    monkeypatch.setattr(owf, "oracle_agent", AsyncMock(return_value="storyteller_gm"))
    # Patch agents_map within oracle_workflow
    agents_map_patch = owf.agents_map.copy()
    agents_map_patch["storyteller_gm"] = fake_storyteller # Map action to the fake workflow
    monkeypatch.setattr(owf, "agents_map", agents_map_patch)

    # Patch necessary cl functions within oracle_workflow
    monkeypatch.setattr(owf.cl, "user_session", MagicMock(get=lambda k,d=None: {}))
    monkeypatch.setattr(owf, "append_log", lambda *a, **kw: None)
    monkeypatch.setattr(owf, "persona_classifier_agent", AsyncMock(return_value={"persona": "storyteller_gm"})) # Assume classifier runs

    state = ChatState(
        messages=[HumanMessage(content="Tell me a story", name="Player")],
        thread_id="t",
        current_persona="storyteller_gm",
    )
    # Pass state object as second argument
    result_state = await oracle_workflow(
        {"messages": state.messages, "previous": state},
        state, # Pass state object
        config={"configurable": {"thread_id": "t"}}
    )

    assert isinstance(result_state, ChatState), f"Expected ChatState, got {type(result_state)}"
    contents = [m.content for m in result_state.messages]
    assert "Lore info" in contents, f"Lore info missing in {contents}"
    assert "Story continues" in contents, f"Story continues missing in {contents}"
