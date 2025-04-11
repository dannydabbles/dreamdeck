# Dreamdeck Development Roadmap: Oracle-Centric Refactor

This `plan.md` outlines a phased implementation plan for refactoring Dreamdeck to an Oracle-centric architecture, where a central Oracle agent decides the flow of actions within each user turn.
Each phase is designed to be a reasonable unit of work.
**Only proceed to the next phase after the previous one is fully implemented and tested.**

---

## Phase 0: Fix Test Environment & Chainlit Context Issues

**Goal:** Get the test suite passing by resolving Chainlit import/context errors. This is crucial before making architectural changes.

**Tasks:**
1.  **Identify Failing Tests:** Run `make test` and pinpoint tests failing due to Chainlit errors (e.g., `AttributeError: 'NoneType' object has no attribute 'session'`, issues with `cl.Message`, `cl.user_session`, `cl.step`).
2.  **Review Mocking Fixtures:** Examine `tests/conftest.py` and fixtures like `mock_chainlit_context` and `mock_cl_environment` in integration tests.
3.  **Apply/Enhance Mocking:**
    *   Ensure relevant fixtures are applied (`@pytest.mark.usefixtures(...)` or `autouse=True`).
    *   Enhance fixtures to mock *all* necessary Chainlit components: `cl.user_session.get/set`, `cl.Message` (class and methods like `.send()`, `.stream_token()`, `.update()`, `.id`), `cl.step` decorator (replace with no-op), `cl.context.session.thread_id`, `cl.AsyncLangchainCallbackHandler`.
    *   Use examples from `tests/integration/test_event_handlers.py` and `tests/smoke/test_commands.py`.
4.  **Verify:** Run `make test` repeatedly until Chainlit-related errors are resolved.

---

## Phase 1: Implement the Oracle Agent and Decision Loop [COMPLETED]

**Goal:** Create the core Oracle agent and refactor `oracle_workflow.py` to use an iterative decision loop driven by the Oracle.

**Status:** Implemented.
*   Created `src/agents/oracle_agent.py` which uses `oracle_decision_prompt.j2`.
*   Refactored `src/oracle_workflow.py` to use a `while` loop driven by `oracle_agent`.
*   Removed `director_agent`.
*   Updated `src/agents/__init__.py` to include persona agents in `agents_map`.
*   Added `tool_results_this_turn` to `ChatState` as a placeholder for Phase 3.

**Notes for Next Phase:**
*   Phase 2 needs to simplify the persona workflows in `src/persona_workflows.py` as they are now only called at the end of the Oracle loop. They should primarily contain the final response generation logic (like `_generate_story`).
*   Tool agents need review to ensure they *don't* call the writer agent themselves.
*   The handling of agent output (list vs dict vs ChatState) in `oracle_workflow.py` needs standardization in Phase 2.
*   Metadata patching (`type`, `persona`, `agent`) was added in the loop but needs review/refinement in Phase 2.
*   The `oracle_decision_prompt.j2` currently has a placeholder for `tool_results_this_turn`; Phase 3 will fully integrate this.

---

## Phase 2: Simplify Persona Workflows & Agent Integration

**Goal:** Adapt persona workflows and tool agents to fit the new Oracle loop.

**Tasks:**
1.  **Refactor `src/persona_workflows.py`:**
    *   Simplify main workflow functions (e.g., `storyteller_workflow`). They are now only called at the *end* of the Oracle loop.
    *   Remove internal tool calls. They should primarily call their core logic (e.g., `_generate_story`) with the final `state`.
    *   Ensure these simplified functions are correctly mapped in `src/agents/__init__.py:agents_map`.
2.  **Review Tool Agents:**
    *   Ensure all tool agents (`dice_agent`, `web_search_agent`, etc.) accept `state`, return `AIMessage` list, and *do not* automatically call the writer agent.

---

## Phase 3: Refine State Management for Oracle

**Goal:** Ensure the Oracle receives necessary intermediate results from within the *same turn*.

**Tasks:**
1.  **Modify `src/models.py` (`ChatState`):**
    *   Add `tool_results_this_turn: List[BaseMessage] = Field(default_factory=list)`.
2.  **Update `src/oracle_workflow.py`:**
    *   Clear `state.tool_results_this_turn` before the loop.
    *   Append successful *tool* agent outputs to `state.tool_results_this_turn` inside the loop.
3.  **Update `src/prompts/oracle_decision_prompt.j2`:**
    *   Add `tool_results_this_turn` to the context.
    *   Update rules/examples explaining its use.

---

## Phase 4: Integrate Persona Classification

**Goal:** Decide how and when persona classification happens within the Oracle flow.

**Tasks:**
1.  **Modify `src/oracle_workflow.py`:**
    *   Call `persona_classifier_agent` *once* at the beginning of the `oracle_workflow` function (before the loop).
    *   Update `state.current_persona` based on the result (respecting suppression/confirmation logic if kept).
    *   Ensure the `current_persona` is passed as context to the `oracle_agent`'s prompt.

---

## Phase 5: Comprehensive Integration Testing

**Goal:** Verify the new Oracle-driven flow works correctly for various scenarios.

**Tasks:**
1.  **Update/Create Integration Tests:** Focus on `tests/integration/test_oracle_workflow_integration.py` or similar.
2.  **Mock LLM Calls:** Mock `oracle_agent`, tool agents, persona agents, and `persona_classifier_agent`.
3.  **Test Scenarios:**
    *   Simple Turn (Tool -> Persona).
    *   Multi-Tool Turn (Tool A -> Tool B -> Persona).
    *   Direct Persona Turn.
    *   Max Iterations Hit.
    *   Persona Switch Flow.
    *   Error Handling (Tool failure, Oracle failure).
4.  **Assert State:** Check final `ChatState.messages` for correct sequence and content.

---

## Phase 6: Cleanup and Documentation

**Goal:** Remove obsolete code and update documentation.

**Tasks:**
1.  **Remove Redundancy:** Delete `src/agents/director_agent.py` and its prompt if fully replaced. Update imports. Remove old workflow logic.
2.  **Update `plan.md`:** Mark these phases as complete.
3.  **Update `README.md` / Docs:** Explain the new Oracle architecture, the role of `oracle_decision_prompt.j2`, and the turn lifecycle.

---

# Notes

- **Always run `make test` after each phase.**
- **Fix any test failures before proceeding.**
- **Keep commits small and focused per phase.**
- **Add comments explaining any tricky logic.**

---

# End of Plan
