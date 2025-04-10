# **Multi-Persona Modular Workflow Refactor & CLI Roadmap**

This roadmap guides an LLM through incremental, manageable phases to refactor the application for multi-persona workflows, CLI support, persistent state, and improved modularity.

---

## **Phase 1: Modularize Persona Workflows**

**Status:** ✅ Completed

- Implemented in `src/persona_workflows.py`
- Contains async functions per persona, no Chainlit UI calls
- Exports `persona_workflows` registry
- Ready for integration in Phase 2

**Goal:**  
Isolate each persona’s logic into a standalone async function or LangGraph workflow, decoupled from Chainlit or UI code.

**Tasks:**  
1. For each persona (e.g., `therapist`, `storyteller_gm`, `secretary`), create a function:  
   ```python
   async def therapist_workflow(inputs: dict, state: ChatState) -> list[BaseMessage]:
       # Persona-specific logic here
   ```
2. Move persona-specific prompts, tool calls, and logic into these functions.  
3. Avoid any Chainlit UI calls (`cl.Message`, etc.) inside these workflows.  
4. Export these workflows in a new or existing module, e.g., `src/persona_workflows.py`.  
5. Create a **persona registry** dictionary:  
   ```python
   persona_workflows = {
       "therapist": therapist_workflow,
       "storyteller_gm": storyteller_workflow,
       "secretary": secretary_workflow,
       # ...
   }
   ```

**Notes:**  
- Use existing code in `src/agents/` as a base.  
- Pass `ChatState` and `inputs` explicitly.  
- These workflows will be invoked by the oracle and CLI.

---

**Next:** Implement `oracle_workflow()` that dispatches to these persona workflows based on current persona or classifier output.

---

## **Phase 2: Implement Oracle Workflow**

**Status:** ✅ Completed

- Implemented in `src/oracle_workflow.py`
- `oracle_workflow()` dispatches to persona workflows based on current persona
- Calls persona classifier if persona unset or forced
- Replaces old `chat_workflow` as the main entrypoint
- Integrated transparently into event handlers

**Goal:**  
Create a central async function that routes inputs to the correct persona workflow, handles persona switching, and manages errors.

**Notes:**  
- The oracle now cleanly dispatches to modular persona workflows.
- Persona switching is handled via classifier or explicit override.
- Next, CLI can invoke `oracle_workflow()` directly.

## **Phase 3: CLI Interface for Persona Invocation (using `click`)**

**Goal:**  
Add a CLI tool to invoke persona workflows directly from the command line, implemented with the `click` package.

**Tasks:**  
1. Create a CLI app file, e.g., `cli.py`.  
2. Use **`click`** decorators to define commands and options.  
3. Implement a `chat` command:  
   ```bash
   python cli.py chat --persona therapist --input "Hello"
   ```  
4. The CLI should:  
   - Load or create a `ChatState` (optionally from a file).  
   - Call `oracle_workflow()` with the input and state.  
   - Print the persona’s response(s).  
   - Optionally save updated state back to file.  
5. Add commands:  
   - `list-personas` (list available personas)  
   - `switch-persona` (change persona mid-session)  
   - `export` (save chat history to markdown or JSON)

**Notes:**  
- CLI should **reuse the same workflows** as the server.  
- Avoid duplicating logic.  
- This enables automation, testing, and scripting.  
- `click` provides a clean, user-friendly CLI interface.

---

## **Phase 4: Persistent Chat State & File Storage Helpers**

**Goal:**  
Enable saving/loading of chat state and structured file storage for persona and shared data.

**Tasks:**  
1. Implement helpers in a new module, e.g., `src/storage.py`:  
   - `save_state(state: ChatState, path: str)`  
   - `load_state(path: str) -> ChatState`  
2. Add directory helpers:  
   - `get_shared_daily_dir(date: str) -> Path`  
   - `get_persona_daily_dir(persona: str, date: str) -> Path`  
3. Update persona workflows to:  
   - Read from shared daily directory (e.g., news, events).  
   - Read/write persona-specific daily files (e.g., memories, notes).  
4. Ensure directories are created if missing.  
5. Optionally, update CLI and oracle to save state after each interaction.

**Notes:**  
- Use ISO date format: `YYYY-MM-DD`.  
- This supports future cron jobs and data ingestion.  
- Enables session persistence and auditability.  
- **Persist ChromaDB vector store data across sessions:**  
  - When saving chat state, ensure relevant embeddings/documents are stored in ChromaDB’s persistent database.  
  - When resuming a conversation, reload or query ChromaDB as needed to restore context.  
  - Do **not** clear or reset ChromaDB between sessions.  
  - This ensures knowledge, memories, and embeddings remain available across conversation reloads.

---

## **Phase 5: Persona Switching, Tool Preferences, and Prompt Customization**

**Goal:**  
Enhance persona experience with explicit switching, tailored prompts, and tool filtering.

**Tasks:**  
1. Add `/persona` slash command (server) and CLI flag to **explicitly switch personas**.  
2. When switching, update `state.current_persona` and log the switch.  
3. Load **persona-specific prompt templates** dynamically from `prompts/`.  
4. Enforce `PERSONA_TOOL_PREFERENCES` to filter or prioritize tools per persona.  
5. Add onboarding message per persona (e.g., "Hi, I'm your therapist").  
6. Log persona switches and tool invocations to daily directories.

**Notes:**  
- Improves immersion and control.  
- Keeps persona behavior consistent.  
- Logging aids debugging and analytics.

---

## **Phase 6: Error Handling, Logging, and Memory Summarization**

**Goal:**  
Make the system more robust, transparent, and efficient.

**Tasks:**  
1. Add try/except blocks in oracle and persona workflows.  
2. On error, return a friendly message and log details.  
3. Log all persona switches, tool calls, and errors with timestamps.  
4. Periodically summarize long chat histories into concise memories.  
5. Store summaries in persona daily directories.  
6. Use summaries to prime prompts instead of full history when appropriate.

**Notes:**  
- Improves reliability and user experience.  
- Keeps context manageable for LLMs.  
- Facilitates future analysis.

---

## **General Tips for the LLM**

- **Always pass and update `ChatState`** to maintain context.  
- **Avoid UI-specific code** inside persona workflows.  
- **Use async/await** consistently.  
- **Keep code modular and DRY** (don’t repeat yourself).  
- **Add docstrings and comments** for clarity.  
- **Test each phase incrementally** before moving on.  
- **Log important events** for debugging and audit.

---

## **Phase 7: Organize Tests into Smoke and Integration Suites**

**Goal:**  
Separate fast, unit-level smoke tests from slower, integration and long-running tests.  
Ensure prompt templates and multi-agent workflows are properly validated.

**Tasks:**  
1. **Create two test directories:**  
   - `tests/smoke/` for **fast, isolated unit tests**  
   - `tests/integration/` for **longer, multi-component, or LLM-involving tests**  
2. **Move existing fast tests** (e.g., simple function/unit tests) into `tests/smoke/`.  
3. **Move or write integration tests** into `tests/integration/`, including:  
   - **Prompt template validation:**  
     - Load each persona’s prompt template.  
     - Run a dummy LLM call (mocked or real) to ensure no syntax errors.  
   - **End-to-end persona workflow tests:**  
     - Simulate multi-turn conversations.  
     - Check persona switching works.  
     - Validate persistence and reload.  
   - **ChromaDB persistence tests:**  
     - Add/query embeddings.  
     - Restart and verify data is retained.  
4. **Update your `Makefile`:**  
   - `make smoke` runs only fast tests:  
     ```bash
     pytest tests/smoke
     ```  
   - `make integration` runs only integration tests:  
     ```bash
     pytest tests/integration
     ```  
   - `make test` runs **both** suites:  
     ```bash
     pytest tests/smoke tests/integration
     ```  
5. **Add a README or comments** explaining the difference and when to run each suite.  
6. **Encourage running smoke tests frequently** during development, and integration tests before merges or releases.

**Notes:**  
- Integration tests **may call real LLM endpoints** or use **high-fidelity mocks**.  
- Prompt template tests **catch errors early** in persona-specific prompts.  
- This separation **keeps CI fast** while ensuring **deep validation** is still performed regularly.  
- You can later add **nightly or scheduled runs** for integration tests if desired.

---

## **End of Roadmap**

Proceed **one phase at a time**.  
After each phase, **test thoroughly** before starting the next.  
This plan ensures a clean, modular, extensible, and robust multi-persona conversational system, with **properly organized testing** to maintain quality.

---
