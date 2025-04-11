# Dreamdeck Refactor Roadmap: Migrating to langgraph-supervisor

This document is a step-by-step roadmap for refactoring Dreamdeck to use a hierarchical multi-agent orchestration pattern inspired by langgraph-supervisor. Each phase is designed to be a reasonable, self-contained chunk of work for a coding LLM. Follow the phases in order. Each phase includes all context, tips, and requirements needed for implementation.

**All necessary documentation, code snippets, and design patterns from langgraph-supervisor are included below. No external search or package documentation is required.**

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

- **Tool Design Pattern:**  
  Each tool should be a stateless, async function that takes in user input, chat history, and any needed context, and returns a summarized/contextualized output.  
  Example:
  ```python
  from langchain_core.messages import AIMessage
  from langgraph.func import task

  @task
  async def my_tool(state, **kwargs):
      # Compose prompt using state (chat history, user input, etc)
      prompt = f"Summarize: {state.get_recent_history_str()}"
      # Call LLM (use your preferred LLM interface)
      response = await llm.ainvoke([("system", prompt)])
      return [AIMessage(content=response.content, name="my_tool")]
  ```
- **Handoff Mechanism:**  
  Tools can be called by agents or the supervisor. When a tool is called, its output is appended to the conversation state and can be used by subsequent agents.

- **Statelessness:**  
  Tools should not manage persona logic or global state. They should only process their input and return output.

- **Testing:**  
  Each tool should have isolated tests that check its output given a mock state.

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

- **Agent Design Pattern:**  
  Each persona agent is an async function or callable class that takes in the conversation state and returns a list of messages (usually a single AIMessage).  
  Example:
  ```python
  from langchain_core.messages import AIMessage
  from langgraph.func import task

  @task
  async def persona_agent(state, **kwargs):
      # Use a persona-specific prompt
      prompt = f"You are a {state.current_persona}. Respond to the user."
      response = await llm.ainvoke([("system", prompt)])
      return [AIMessage(content=response.content, name=state.current_persona)]
  ```
- **Persona Prompts:**  
  Each agent should use a Jinja2 prompt template specific to its persona, filled with recent chat, memories, tool results, and user preferences.

- **Tool Calls:**  
  Persona agents can call tools by returning a special "handoff" message or by using a supervisor routing mechanism (see Phase 5).

- **No Global State:**  
  Agents should only manage their own prompt/context, not global state.

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

- **Supervisor Design Pattern:**  
  The supervisor is a central async function or class that receives user input and conversation state, decides which agent or tool to invoke next, and manages handoff and message history.

  Example supervisor logic:
  ```python
  async def supervisor(state, **kwargs):
      # Decide which agent or tool to call based on user input or context
      if "roll" in state.get_last_human_message().content:
          return await dice_tool(state)
      elif "search" in state.get_last_human_message().content:
          return await search_tool(state)
      else:
          # Default: route to current persona agent
          persona = state.current_persona
          return await persona_agents[persona](state)
  ```

- **Handoff Tools:**  
  The supervisor can use "handoff tools" to pass control to a specific agent or tool.  
  Example handoff tool:
  ```python
  from langchain_core.tools import tool
  from langchain_core.messages import ToolMessage
  from langgraph.types import Command

  @tool("handoff_to_agent", description="Assign task to a specific agent")
  def handoff_to_agent(agent_name: str, state: dict, tool_call_id: str):
      tool_message = ToolMessage(
          content=f"Transferred to {agent_name}",
          name="handoff_to_agent",
          tool_call_id=tool_call_id,
      )
      messages = state["messages"]
      return Command(
          goto=agent_name,
          graph=Command.PARENT,
          update={
              "messages": messages + [tool_message],
              "active_agent": agent_name,
          },
      )
  ```

- **Message History Management:**  
  The supervisor manages what message history is passed to each agent/tool. You can choose to pass the full history, only the last message, or a custom slice.

- **Memory:**  
  Use a memory/checkpointing mechanism to persist state if needed.  
  Example:
  ```python
  from langgraph.checkpoint.memory import InMemorySaver
  checkpointer = InMemorySaver()
  app = workflow.compile(checkpointer=checkpointer)
  ```

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
