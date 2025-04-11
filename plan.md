# Dreamdeck Development Roadmap

This `plan.md` outlines a phased implementation plan for extending and improving Dreamdeck.  
Each phase is designed to be a reasonable unit of work for an LLM to complete in one shot.  
**Only proceed to the next phase after the previous one is fully implemented and tested.**

---

## Phase 1: Improve Persona Metadata Consistency

**Phase 1 completed:** All persona workflows now enforce `"type": "ai"` and `"persona"` metadata on AI messages.  
This improves downstream vector store indexing and test reliability.  
Next, proceed with **Phase 2**: vector store `.put()` metadata consistency.

---

## Phase 2: Enhance Vector Store Metadata Handling

**Phase 2 completed:** All vector store `.put()` calls for AI messages now enforce:
- `"type": "ai"`
- `"persona"` (prefer message metadata, fallback to current persona)
- `"author"` (AI message name)

This ensures consistent metadata for search, filtering, and persona tracking.

Next, proceed with **Phase 3**: Add more realistic multi-tool conversation tests.

---

## Phase 3: Add More Realistic Multi-Tool Conversation Tests

**Phase 3 completed:** Added integration test simulating a multi-tool turn with persona switch, multiple tool calls, and final story generation.  
All tool outputs and metadata are verified.  
Next, proceed with **Phase 4**: Improve persona switching logic and tests.

---

## Phase 4: Improve Persona Switching Logic and Tests

**Phase 4 completed:**  
Persona switching now suppresses repeated prompts after user declines, gracefully falls back on classifier errors, and is covered by new tests for these cases and for forcible persona switching.

Next, proceed with **Phase 5**: Documentation and Developer Notes.

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
