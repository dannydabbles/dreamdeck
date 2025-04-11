# Dreamdeck Refactor Roadmap: Migrating to langgraph-supervisor

This document is a step-by-step roadmap for refactoring Dreamdeck to use [langgraph-supervisor](https://github.com/langchain-ai/langgraph-supervisor-py) for hierarchical multi-agent orchestration. Each phase is designed to be a reasonable, self-contained chunk of work for a coding LLM. Follow the phases in order. Each phase includes all context, tips, and requirements needed for implementation.

---

## **Phase 1: Project Cleanup and Preparation**

**Goal:** Remove legacy workflow code and tests to reduce context and avoid confusion.

**Tasks:**
- Archive or delete the following files (if not already done):
  - `src/oracle_workflow.py`
  - `src/agents/oracle_agent.py`
  - `src/agents/director_agent.py`
  - `src/persona_workflows.py`
  - `src/agents/__init__.py`
  - `src/tools_and_agents.py`
  - `cli.py`
  - `tests/integration/test_oracle_workflow_integration.py`
  - `tests/integration/test_persona_workflow.py`
  - `tests/integration/test_persona_workflows.py`
  - `tests/integration/test_persona_system.py`
  - `tests/integration/test_workflows.py`
  - `tests/smoke/test_director_agent.py`
  - Archive or remove Jinja2 prompts only used by the above:
    - `src/prompts/director_prompt.j2`
    - `src/prompts/oracle_decision_prompt.j2` (add this file to the chat if you want to archive it)
- Commit the changes with a clear message.

**Tips:**
- Use `git mv` to archive, or `rm` to delete.
- Run `make test` to ensure remaining tests pass.

---

## **Phase 2: Inventory and Define Agents & Tools**

**Goal:** Clearly define which components are persona agents and which are tools.

**Tasks:**
- List all current "agents" and "tools" in the codebase.
- For each, decide:
  - Is it a **persona agent** (LLM with a persona prompt, e.g. Storyteller GM, Therapist, Secretary, etc.)?
  - Is it a **tool** (atomic, stateless, LLM-backed function, e.g. roll, search, todo)?
- Document the mapping in a table in this file for reference.

**Example Table:**

| Name            | Type    | Notes                                 |
|-----------------|---------|---------------------------------------|
| roll            | Tool    | LLM-backed dice roller                |
| search          | Tool    | LLM-backed web search summarizer      |
| todo            | Tool    | LLM-backed todo manager               |
| writer/GM       | Agent   | Persona agent (Storyteller GM)        |
| therapist       | Agent   | Persona agent                         |
| secretary       | Agent   | Persona agent                         |
| coder           | Agent   | Persona agent                         |
| friend          | Agent   | Persona agent                         |
| lorekeeper      | Agent   | Persona agent                         |
| dungeon_master  | Agent   | Persona agent                         |

**Tips:**
- Tools should be stateless and not manage persona logic.
- Persona agents should use Jinja2 prompts for their persona.

---

## **Phase 3: Refactor Tools as LLM-Backed Functions**

**Goal:** Implement all tools as stateless, LLM-backed functions compatible with langgraph-supervisor.

**Tasks:**
- For each tool (roll, search, todo, etc.):
  - Refactor as a function that takes in user input, chat history, and any needed context.
  - The function should call an LLM (using a Jinja2 prompt) and return a summarized/contextualized output.
  - Use the langgraph-supervisor tool interface.
- Remove any persona or state management from tools.
- Update or create tests for each tool.

**Tips:**
- See [langgraph-supervisor tool docs](https://github.com/langchain-ai/langgraph-supervisor-py#customizing-handoff-tools).
- Each tool should be a pure function from input/context to output.

---

## **Phase 4: Refactor Persona Agents**

**Goal:** Implement each persona agent as an LLM-backed agent using langgraph-supervisor.

**Tasks:**
- For each persona agent:
  - Refactor as an agent with a persona-specific Jinja2 prompt.
  - The agent should be able to call tools via the supervisor’s handoff mechanism.
  - Remove any custom workflow logic (e.g. persona_workflows.py).
- Update or create tests for each persona agent.

**Tips:**
- Use the [langgraph-supervisor agent interface](https://github.com/langchain-ai/langgraph-supervisor-py#quickstart).
- Each agent should only manage its own prompt/context, not global state.

---

## **Phase 5: Implement Supervisor Workflow**

**Goal:** Replace custom Oracle/Director logic with a langgraph-supervisor supervisor agent.

**Tasks:**
- Implement a supervisor agent that:
  - Receives user input and conversation state.
  - Decides which agent or tool to invoke next (using a prompt, rules, or LLM).
  - Handles handoff and message history as per langgraph-supervisor conventions.
- Wire up the supervisor to the Chainlit/chat interface.
  - User input goes to the supervisor, which routes to the correct agent/tool.
  - Supervisor manages message history and persona switching.
- Update or create tests for the new workflow.

**Tips:**
- See [langgraph-supervisor quickstart](https://github.com/langchain-ai/langgraph-supervisor-py#quickstart).
- Use supervisor’s built-in memory and message management.

---

## **Phase 6: Final Integration and Cleanup**

**Goal:** Remove obsolete code, update documentation, and ensure all tests pass.

**Tasks:**
- Remove any remaining obsolete code or references to the old workflow.
- Update CLI and Chainlit event handlers to use the new supervisor workflow.
- Update documentation to reflect the new architecture.
- Run and fix all tests to ensure correctness.

**Tips:**
- Run `make test` to verify.
- Update README and any user-facing docs.

---

## **General Notes and Tips**

- **Prompts:** Most Jinja2 prompts will still be needed. Only remove those tied exclusively to deleted code.
- **Testing:** Update or rewrite tests as you refactor each phase.
- **Supervisor:** Let the supervisor handle all routing and persona switching.
- **Statelessness:** Tools should be stateless; agents should only manage their own context.
- **Documentation:** Keep this plan.md updated as you progress.

---

**Proceed phase by phase. After each phase, verify tests and code health before moving to the next.**
