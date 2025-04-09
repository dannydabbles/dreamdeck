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

**Status: Completed**

**Goal:** Remove the legacy `decision_agent` and solidify the `orchestrator_agent` as the sole routing mechanism. Optionally rename for clarity.

**Tasks:**

1.  **Delete Decision Agent:**
    *   Delete `src/agents/decision_agent.py`.
    *   Delete `src/prompts/decision_prompt.j2`.
    *   Delete `tests/test_decision_agent.py`.
    *   **DONE**
2.  **Refactor Workflow:**
    *   Ensure `src/workflows.py` (`_chat_workflow`) relies *only* on the output of `orchestrator_agent` for routing decisions. Remove any fallback logic that might have implicitly relied on the old decision agent structure.
    *   **DONE** (Workflow already relied solely on orchestrator output)
3.  **Rename Orchestrator (Optional but Recommended):**
    *   Rename `src/agents/orchestrator_agent.py` to `src/agents/director_agent.py`. **DONE**
    *   Rename the functions (`_decide_actions` -> `_direct_actions`, `orchestrate` -> `direct`, `orchestrator_agent` -> `director_agent`).
    *   Update imports and calls in `src/workflows.py`, `src/agents/__init__.py`.
    *   Rename `src/prompts/orchestrator_prompt.j2` to `src/prompts/director_prompt.j2` and update `config.yaml`.
    *   Rename `tests/test_orchestrator_agent.py` to `tests/test_director_agent.py` and update tests.

**Rationale:** Streamlines the control flow, removes ambiguity between decision/orchestration, and prepares for potentially different orchestrators per persona. Depends on Phase 2.

**Notes for Next Phase (Phase 4):**
*   Phase 3 successfully removed the legacy `decision_agent` and renamed the `orchestrator_agent` to `director_agent` for clarity.
*   All relevant files, prompts, tests, and configuration have been updated to reflect this change.
*   The workflow now clearly uses the `director_agent` as the single point of routing.
*   Phase 4 involves updating vector store persistence to include the current persona in message metadata.
---

## Phase 4: Memory & Persistence Updates (Tagging with Persona)

**Status: Completed**

**Goal:** Ensure the active persona is stored alongside messages in the vector store and update TODO file paths if necessary.

**Tasks:**

1.  **Update Vector Store `put` Calls:**
    *   All vector store `.put()` calls in `src/event_handlers.py`, `src/workflows.py`, and `src/commands.py` now include the current persona in metadata: `metadata={"type": "...", "author": "...", "persona": state.current_persona}`.
2.  **Update TODO Agent File Paths:**
    *   The TODO agent now saves files under persona-specific subdirectories: `helper/<persona>/<date>/todo.md`.

**Rationale:** Enriches persisted data with persona context, enabling future persona-specific memory retrieval or analysis.

---

**Notes for Next Phase (Phase 5):**

* Persona metadata is now consistently stored with all vector store messages.
* TODO files are saved per persona, improving organization.
* Next, focus on UI improvements with Chainlit Steps and Avatars.

---

## Phase 5: UI Enhancements - Steps & Avatars

**Status: Completed**

**Goal:** Improve UI feedback and visual appeal using Chainlit's Steps and Avatars.

**Tasks:**

1.  **Add `@cl.step` Decorators:**
    *   Added `@cl.step(name="Descriptive Name", type="tool")` decorators to the core async functions within agent files:
        *   `src/agents/writer_agent.py` (`_generate_story`)
        *   `src/agents/knowledge_agent.py` (`_knowledge`)
        *   `src/agents/dice_agent.py` (`_dice_roll`)
        *   `src/agents/web_search_agent.py` (`_web_search`)
        *   `src/agents/todo_agent.py` (`_manage_todo`)
        *   `src/agents/storyboard_editor_agent.py` (`_generate_storyboard`)
    *   Step names are informative and appear in the Chainlit UI during execution.

2.  **Avatars:**
    *   No explicit avatar customization was implemented in this phase.
    *   Can be added later if desired.

**Rationale:** Improves transparency of agent execution in the UI, making it easier to follow the workflow.

---

### Notes for Next Phase (Phase 6):

* The UI now clearly shows which agent/tool is running via Chainlit steps.
* Optional: Customize avatars or add more detailed step descriptions.
* Next, focus on CLI interface for agent testing and development.
