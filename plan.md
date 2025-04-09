# Dreamdeck Development Roadmap

This document outlines the planned phases for enhancing Dreamdeck, focusing on improving player experience, developer experience, and code maintainability by leveraging Chainlit features and refactoring the agent architecture.

**Relevant Chainlit Documentation:**

*   Chat Profiles: <https://docs.chainlit.io/advanced-features/chat-profiles>
*   Chat Settings: <https://docs.chainlit.io/advanced-features/chat-settings>
*   Chat Lifecycle: <https://docs.chainlit.io/concepts/chat-lifecycle>
*   User Session: <https://docs.chainlit.io/concepts/user-session>
*   Steps (Chain of Thought): <https://docs.chainlit.io/concepts/step>
*   Avatars: <https://docs.chainlit.io/customisation/avatars>
*   LangGraph Integration: <https://docs.chainlit.io/integrations/langchain> (Relevant for workflow understanding)

---

## Phase 1: Foundation - Chat Profiles & Persona State

**Status: Completed**

**Goal:** Introduce distinct application modes (personas) using Chainlit Profiles and manage the active persona within the application state.

**Tasks:**

1.  **Define Chat Profiles:**
    *   In `src/app.py`, define at least two `@cl.profile` functions (e.g., "Storyteller GM", "Note-Taking PA").
    *   These profiles should set an initial value in `cl.user_session` indicating the selected persona (e.g., `cl.user_session.set("current_persona", "Storyteller GM")`).
    *   **DONE**
2.  **Update Configuration:**
    *   In `config.yaml`, add sections for persona-specific configurations (e.g., different system prompts for the writer agent under `agents.writer_agent.personas.Storyteller GM.prompt_key`).
    *   Update `src/config.py` to load and potentially structure these persona-specific settings.
    *   **DONE** (Added `personas` section in `config.yaml`, `src/config.py` loads it automatically. Accessing via `.dict()` in agent for now.)
3.  **Update Chat State:**
    *   In `src/models.py`, add `current_persona: str = "default"` (or similar) to the `ChatState` model.
    *   **DONE**
4.  **Integrate Persona into Lifecycle:**
    *   In `src/event_handlers.py` (`on_chat_start`): If a profile is active, retrieve the persona name set by the profile function and store it in the initial `ChatState.current_persona`. If no profile is active, use a default value.
    *   In `src/event_handlers.py` (`on_chat_resume`): Attempt to load the `current_persona` from thread tags (if `auto_tag_thread` is enabled in `.chainlit/config.toml`) or from stored metadata. Fallback to a default if not found.
    *   **DONE** (Implemented retrieval from session in `on_chat_start` and from tags in `on_chat_resume`.)
5.  **Update Writer Agent:**
    *   Modify `src/agents/writer_agent.py` (`_generate_story`) to load the appropriate system prompt based on `state.current_persona`.
    *   **DONE** (Loads prompt key from config based on persona, then retrieves prompt text.)

**Rationale:** Establishes the core mechanism for different application modes, essential for subsequent agent and workflow modifications.

---

**Notes for Next Phase (Phase 2):**
*   Phase 1 successfully introduced personas and linked them to writer prompts.
*   The `knowledge_agent` in Phase 2 should also be made persona-aware, potentially loading different base prompts or using different LLM settings (temp, tokens) based on `state.current_persona`. This will require adding persona configurations for the knowledge agent in `config.yaml` similar to how it was done for the writer agent.
*   The orchestrator prompt (`orchestrator_prompt.j2`) and logic (`orchestrator_agent.py`) will need modification to output the new `{"action": "knowledge", "type": "..."}` format.
*   The workflow (`workflows.py`) will need to be updated to correctly call the `knowledge_agent` with the `knowledge_type` parameter when the orchestrator requests it.

---

## Phase 2: Agent Consolidation - Knowledge Agent

**Status: Completed**

**Goal:** Reduce code duplication by consolidating `character_agent`, `lore_agent`, and `puzzle_agent` into a single `knowledge_agent`.

**Tasks:**

1.  **Create Knowledge Agent:**
    *   Create `src/agents/knowledge_agent.py`.
    *   Define `async def _knowledge(state: ChatState, knowledge_type: str)` and a `@task` wrapper `knowledge_agent`.
    *   The agent should load the correct prompt file (e.g., `character_prompt.j2`, `lore_prompt.j2`) from `src/prompts/` based on the `knowledge_type` argument.
    *   It should use persona-specific settings (temp, endpoint, max_tokens) from `config.yaml` if defined, falling back to defaults.
    *   **DONE** (`src/agents/knowledge_agent.py` created)
2.  **Update Orchestrator:**
    *   Modify `src/agents/orchestrator_agent.py` (`_decide_actions`) and its prompt (`src/prompts/orchestrator_prompt.j2`).
    *   The orchestrator now outputs actions as a list which can contain strings (like `"search"`) or dictionaries (like `{"action": "knowledge", "type": "character"}`).
    *   **DONE**
3.  **Update Workflow:**
    *   Modify `src/workflows.py` (`_chat_workflow`) to handle the new orchestrator output format. When `"action": "knowledge"` is received, call `knowledge_agent` passing `state` and the `knowledge_type`.
    *   **DONE**
4.  **Update Agent Map:**
    *   Modify `src/agents/__init__.py` to update `agents_map`. The key might be "knowledge", mapping to `knowledge_agent`. The workflow will need to handle passing the `knowledge_type`.
    *   **DONE** (Removed old agents, added `knowledge_agent` import. The map itself doesn't need a "knowledge" key as the workflow handles the dictionary action format directly.)
5.  **Cleanup:**
    *   Delete `src/agents/character_agent.py`, `src/agents/lore_agent.py`, `src/agents/puzzle_agent.py`.
    *   Delete corresponding prompts if they are fully superseded (or keep if the new agent loads them by name).
    *   Delete `tests/test_agents_misc.py`.
    *   **DONE**
6.  **Update Tests:**
    *   Create `tests/test_knowledge_agent.py` to test the consolidated agent with different `knowledge_type` inputs.
    *   Update `tests/test_orchestrator_agent.py` and `tests/test_workflows.py` to reflect the new action format and agent calls.
    *   **DONE**
7.  **Update UI Settings:**
    *   Remove specific Chat Settings sliders/inputs for character, lore, and puzzle agents from `src/event_handlers.py` (`on_chat_start`) as the new agent reads these dynamically.
    *   **DONE**

**Rationale:** Simplifies the agent architecture, reduces redundancy, and makes adding new "knowledge" types easier in the future. Depends on Phase 1 for persona-specific settings.

**Notes for Next Phase (Phase 3):**
*   Phase 2 successfully consolidated the knowledge-based agents.
*   The orchestrator now correctly outputs a list of actions, including the dictionary format for the knowledge agent.
*   The workflow handles this new format.
*   Persona-specific settings (temp, tokens, endpoint) are read dynamically by the `knowledge_agent` based on keys like `character_temp`, `lore_temp` etc. found in the `cl.user_session.get("chat_settings", {})`. The UI controls for these were removed in `on_chat_start` to avoid clutter; consolidated controls could be added later if needed.
*   Phase 3 involves removing the legacy `decision_agent` entirely and potentially renaming the `orchestrator_agent` to `director_agent`.

---

## Phase 3: Orchestration Refinement & Director Renaming

**Goal:** Remove the legacy `decision_agent` and solidify the `orchestrator_agent` as the sole routing mechanism. Optionally rename for clarity.

**Tasks:**

1.  **Delete Decision Agent:**
    *   Delete `src/agents/decision_agent.py`.
    *   Delete `src/prompts/decision_prompt.j2`.
    *   Delete `tests/test_decision_agent.py`.
2.  **Refactor Workflow:**
    *   Ensure `src/workflows.py` (`_chat_workflow`) relies *only* on the output of `orchestrator_agent` for routing decisions. Remove any fallback logic that might have implicitly relied on the old decision agent structure.
3.  **Rename Orchestrator (Optional but Recommended):**
    *   Rename `src/agents/orchestrator_agent.py` to `src/agents/director_agent.py`.
    *   Rename the functions (`_decide_actions` -> `_direct_actions`, `orchestrate` -> `direct`, `orchestrator_agent` -> `director_agent`).
    *   Update imports and calls in `src/workflows.py`, `src/agents/__init__.py`.
    *   Rename `src/prompts/orchestrator_prompt.j2` to `src/prompts/director_prompt.j2` and update `config.yaml`.
    *   Rename `tests/test_orchestrator_agent.py` to `tests/test_director_agent.py` and update tests.

**Rationale:** Streamlines the control flow, removes ambiguity between decision/orchestration, and prepares for potentially different orchestrators per persona. Depends on Phase 2.

---

## Phase 4: Memory & Persistence Updates (Tagging with Persona)

**Goal:** Ensure the active persona is stored alongside messages in the vector store and update TODO file paths if necessary.

**Tasks:**

1.  **Update Vector Store `put` Calls:**
    *   In all places where `vector_store.put` is called (`src/event_handlers.py:on_message`, `src/workflows.py:_chat_workflow`, `src/commands.py`), modify the `metadata` dictionary to include the current persona: `metadata={"type": "...", "author": "...", "persona": state.current_persona}`.
2.  **Update TODO Agent File Paths (Optional):**
    *   If desired, modify `src/agents/todo_agent.py` (`_manage_todo`) to save TODO files into persona-specific subdirectories (e.g., `os.path.join(TODO_DIR_PATH, state.current_persona, current_date)`). Ensure directory creation handles this.
    *   Update `tests/test_todo_agent.py` accordingly.

**Rationale:** Enriches persisted data with persona context, enabling potential future persona-specific memory retrieval or analysis. Depends on Phase 1.

---

## Phase 5: UI Enhancements - Steps & Avatars

**Goal:** Improve UI feedback and visual appeal using Chainlit's Steps and Avatars.

**Tasks:**

1.  **Add `@cl.step` Decorators:**
    *   Add `@cl.step(name="Descriptive Name", type="tool")` or similar decorators to the core async functions within agent files:
        *   `src/agents/writer_agent.py` (`_generate_story`)
        *   `src/agents/knowledge_agent.py` (`_knowledge`)
        *   `src/agents/dice_agent.py` (`_dice_roll`)
        *   `src/agents/web_search_agent.py` (`_web_search`)
        *   `src/agents/todo_agent.py` (`_manage_todo`)
        *   `src/agents/storyboard_editor_agent.py` (`_generate_storyboard`)
    *   Ensure the step names are informative. Adjust placement if the function is already decorated with `@task`.
2.  **Implement Custom Avatars (Optional):**
    *   Create `src/customisation/avatars.py`.
    *   Define `cl.Avatar` instances for different personas (e.g., "Storyteller GM", "Note-Taking PA") and potentially tools ("Dice Roller", "Web Search"). Use image URLs or local files (placed in `public/`).
    *   In `src/app.py` or `src/event_handlers.py`, ensure these avatars are registered or configured.
    *   Modify `cl.Message` creation calls in agent files (`_dice_roll`, `_web_search`, `_manage_todo`, `_generate_story`) and potentially `commands.py` to use the registered avatar names in the `author` field (e.g., `cl.Message(..., author="Storyteller GM")`).
    *   Update tests if they assert on author names.

**Rationale:** Provides better visibility into agent execution for users and developers, and makes the chat interface more engaging. Best done after agent structure is stable (Post Phase 3).

---

## Phase 6: Command Line Interface (CLI) (Optional)

**Goal:** Provide a non-UI method for interacting with agents, primarily for testing and development.

**Tasks:**

1.  **Create `cli.py`:**
    *   Create a new file `cli.py` at the project root.
    *   Use `argparse` to handle command-line arguments (e.g., `--agent`, `--prompt`, `--persona`, `--state-file`).
    *   Import necessary agent functions (potentially needing `call_` wrappers like `call_writer_agent`), `ChatState`, `config`.
    *   Implement logic to:
        *   Load config.
        *   Initialize or load `ChatState`.
        *   Set `state.current_persona` based on args.
        *   Call the specified agent function directly with the state.
        *   Print the agent's response (likely the `content` of the returned `AIMessage`).
        *   Optionally save the final state.
2.  **Add `call_` Wrappers (If Needed):**
    *   Ensure agents intended for CLI use have simple async wrapper functions (like `call_writer_agent`) that take `state` and return the result, separate from the LangGraph `@task` decorators.
3.  **Add Tests (Optional):**
    *   Create `tests/test_cli.py` to test the command-line interface functionality using `subprocess` or by directly calling the CLI functions.

**Rationale:** Offers a valuable tool for developers to test agent logic without the full web UI stack. Largely independent of UI changes.

---
