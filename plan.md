# Dreamdeck Development Roadmap

This `plan.md` outlines a phased implementation plan for extending and improving Dreamdeck.  
Each phase is designed to be a reasonable unit of work for an LLM to complete in one shot.  
**Only proceed to the next phase after the previous one is fully implemented and tested.**

---

## Phase 1: Improve Persona Metadata Consistency

### Goal
Ensure **all AI-generated messages** in persona workflows **always** include metadata fields:
- `"type": "ai"`
- `"persona": <current_persona>`

This will improve downstream vector store indexing, persona tracking, and test reliability.

### Context & Tips
- Your `src/persona_workflows.py` already **patches** metadata for the **secretary** persona.
- Other persona workflows (storyteller, therapist, coder, friend, lorekeeper, dungeon master, default) **do not** enforce this metadata.
- Some AI messages may lack metadata or have incomplete metadata.
- Tests sometimes fail or become fragile if metadata is missing or inconsistent.

### Tasks
1. **In each persona workflow** (`storyteller_workflow`, `therapist_workflow`, etc.):
   - After calling `_generate_story()` or `knowledge_agent()`, **iterate over the returned list**.
   - For each `AIMessage`:
     - If `metadata` is `None`, set it to an empty dict.
     - Set `metadata["type"] = "ai"`.
     - Set `metadata["persona"] = state.current_persona`.
2. **Return the updated list** as the workflow output.
3. **Add comments** explaining this patching is to ensure consistent metadata for downstream processing and tests.

---

## Phase 2: Enhance Vector Store Metadata Handling

### Goal
Make sure **all vector store `.put()` calls** for AI messages **include**:
- `"type": "ai"`
- `"persona": <persona at message creation time>`
- `"author": <AI message name>`

### Context & Tips
- In `src/event_handlers.py`, after workflows run, AI messages are saved to the vector store.
- Sometimes, metadata is missing or incomplete.
- This causes test failures (e.g., expecting `"type": "ai"` but getting `"human"`).
- Consistent metadata improves search, filtering, and persona tracking.

### Tasks
1. In `src/event_handlers.py`, **inside the loop** that saves AI messages to the vector store:
   - Before calling `vector_memory.put()`, **ensure**:
     - `meta["type"] = "ai"`
     - `meta["persona"] = state.current_persona` (if not already set)
     - `meta["author"] = msg.name`
2. Defensive: if `msg.metadata` exists and contains `"persona"`, prefer that.
3. Add comments explaining the importance of consistent metadata.

---

## Phase 3: Add More Realistic Multi-Tool Conversation Tests

### Goal
Simulate **complex player turns** involving **multiple tool invocations** and **persona switches**.

### Context & Tips
- Your current tests cover:
  - Single tool calls
  - Persona switching
  - Slash commands
- But **multi-hop** flows (e.g., classifier → director → search + dice + todo + writer) are less tested.
- Adding these will increase confidence in complex conversation handling.

### Tasks
1. In `tests/integration/test_persona_system.py`, add a new async test:
   - Simulate a user message that triggers:
     - Classifier suggesting a persona switch
     - Director returning multiple actions: e.g., `["search", "roll", "todo", "write"]`
     - Each tool returns a dummy AI message with correct metadata
     - Writer agent generates a final story segment
2. Patch all relevant agents (`persona_classifier_agent`, `director_agent`, `web_search_agent`, `dice_agent`, `todo_agent`, `writer_agent`) to return dummy responses.
3. After the workflow:
   - Assert that **all tool outputs** and the **final story** are present in the state.
   - Assert that **all AI messages** have `"type": "ai"` and correct `"persona"`.
4. Add comments explaining the test simulates a realistic multi-tool turn.

---

## Phase 4: Improve Persona Switching Logic and Tests

### Goal
Make persona switching **more robust** and **better tested**.

### Context & Tips
- Currently, persona switching is prompted after classifier runs.
- User can accept or decline.
- Sometimes, persona is forcibly switched via slash command.
- Edge cases (e.g., user declines, or classifier errors) need better coverage.

### Tasks
1. In `src/event_handlers.py`:
   - When user **declines** a suggested switch, **store** a flag to **suppress re-suggesting** the same persona for the next N turns.
   - When classifier **errors**, **fallback** to current persona without prompting.
2. In `tests/integration/test_persona_system.py`:
   - Add tests for:
     - User declining a switch, then classifier suggesting same persona again (should **not** prompt again immediately).
     - Classifier error fallback.
     - Forcible switch via `/persona` command.
3. Add comments explaining the logic.

---

## Phase 5: Documentation and Developer Notes

### Goal
Provide clear documentation for future developers and LLMs.

### Tasks
1. Update `README.md`:
   - Explain the **persona system** and how metadata is used.
   - Document the **vector store metadata schema**.
   - Describe the **test philosophy**: simulate realistic multi-tool, multi-persona flows.
2. Add a `docs/persona_system.md`:
   - Explain how persona classification, switching, and workflows interact.
   - Include example metadata for AI and human messages.
3. Add a `docs/testing.md`:
   - Summarize test coverage.
   - Explain how to add new integration tests.
   - Tips for mocking LLMs and agents.

---

# Notes

- **Always run `make test` after each phase.**
- **Fix any test failures before proceeding.**
- **Keep commits small and focused per phase.**
- **Add comments explaining any tricky logic.**

---

# End of Plan

This roadmap will improve metadata consistency, test coverage, and maintainability of Dreamdeck.  
Proceed **one phase at a time**, verifying correctness before moving on.
