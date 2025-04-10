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

**Status:** ✅ Completed

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
- Implemented in `cli.py` with commands: `chat`, `list-personas`, `switch-persona`, `export`.
- Chat state is saved in `chat_state.json` for persistence.
- CLI calls `oracle_workflow()` directly, sharing logic with server.
- Supports persona switching and exporting chat history.

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

**Status:** ✅ Completed

- CLI supports explicit persona switching (`switch-persona` command).
- Onboarding message printed after persona switch.
- Persona switches and tool calls logged to daily directories (`helper/<persona>/<date>/log.txt`).
- Persona-specific prompt templates loaded dynamically from `prompts/`.
- Tool preferences enforced via `PERSONA_TOOL_PREFERENCES`.

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

**Status:** ✅ Completed

- All persona workflows wrapped in try/except, return friendly error messages.
- Errors and persona switches logged with timestamps in daily persona directories.
- Tool calls and slash commands logged similarly.
- Chat histories over 50 messages are summarized (simple truncation for now).
- Summaries saved in `helper/<persona>/<date>/memories.txt`.
- Summaries used in prompts instead of full history, plus recent 20 messages.
- Next: improve summarization quality using LLM calls (future enhancement).

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

**Status:** ✅ Completed

- Tests are now split into `tests/smoke/` (fast unit tests) and `tests/integration/` (multi-component, LLM, or slower tests).
- `Makefile` updated with `smoke`, `integration`, and `test` targets.
- `tests/README.md` explains the difference and usage.
- Developers should run smoke tests frequently, and integration tests before merges/releases.

**Notes:**

- Integration tests may call real LLM endpoints or use mocks.
- Prompt template tests catch errors early.
- This split keeps CI fast while ensuring deep validation.
- Consider adding nightly or scheduled integration test runs in future.

---

## **Next**

- Improve prompt template validation coverage.
- Add more integration tests for new personas and workflows.
- Automate test runs in CI/CD pipeline.

---
