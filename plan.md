# Dreamdeck Refactor Roadmap: Migrating to langgraph-supervisor

This document is a step-by-step roadmap for refactoring Dreamdeck to use a hierarchical multi-agent orchestration pattern inspired by langgraph-supervisor. Each phase is designed to be a reasonable, self-contained chunk of work for a coding LLM. Follow the phases in order. Each phase includes all context, tips, and requirements needed for implementation.

**All necessary documentation, code snippets, and design patterns from langgraph-supervisor are included below. No external search or package documentation is required.**

---

## **Phase 1: Project Cleanup and Preparation** ✅ *Completed*

**Goal:** Remove legacy workflow code and tests to reduce context and avoid confusion.

**Status:**  
✅ All listed legacy files and prompts have been deleted or archived.  
✅ Tests pass after cleanup.

**Notes:**  
- All old Oracle/Director/persona_workflows code and related tests are now removed.
- Jinja2 prompts only used by the old workflow have been deleted.
- The codebase is now ready for Phase 2: inventorying and defining agents & tools.

**Helpful tips for next phases:**  
- The codebase is now much cleaner; you can focus on the new agent/tool structure.
- If you need to reference old workflow logic, check git history.
- Run `make test` after each phase to ensure stability.

---

## **Phase 2: Inventory and Define Agents & Tools** ✅ *Completed*

**Goal:** Clearly define which components are persona agents and which are tools.

**Status:**  
✅ All current agents and tools have been inventoried and classified.  
✅ Mapping table added below.  
✅ Ready for Phase 3.

**Agent/Tool Inventory Table:**

| Name                | Type    | Notes                                                      |
|---------------------|---------|------------------------------------------------------------|
| roll                | Tool    | LLM-backed dice roller (`src/agents/dice_agent.py`)        |
| search              | Tool    | LLM-backed web search summarizer (`src/agents/web_search_agent.py`) |
| todo                | Tool    | LLM-backed todo manager (`src/agents/todo_agent.py`)       |
| report              | Tool    | LLM-backed daily report generator (`src/agents/report_agent.py`) |
| knowledge           | Tool    | LLM-backed knowledge/lore generator (`src/agents/knowledge_agent.py`) |
| storyboard_editor   | Tool    | LLM-backed storyboard image prompt generator (`src/agents/storyboard_editor_agent.py`) |
| writer/GM           | Agent   | Persona agent (Storyteller GM, `src/agents/writer_agent.py`) |
| therapist           | Agent   | Persona agent (Therapist, `src/agents/writer_agent.py`)    |
| secretary           | Agent   | Persona agent (Secretary, `src/agents/writer_agent.py`)    |
| coder               | Agent   | Persona agent (Coder, `src/agents/writer_agent.py`)        |
| friend              | Agent   | Persona agent (Friend, `src/agents/writer_agent.py`)       |
| lorekeeper          | Agent   | Persona agent (Lorekeeper, `src/agents/writer_agent.py`)   |
| dungeon_master      | Agent   | Persona agent (Dungeon Master, `src/agents/writer_agent.py`) |
| default             | Agent   | Default persona agent (`src/agents/writer_agent.py`)       |
| persona_classifier  | Tool    | LLM-backed persona classifier (`src/agents/persona_classifier_agent.py`) |

**Tips:**
- Tools are stateless, LLM-backed functions that do not manage persona logic.
- Persona agents use Jinja2 prompts for their persona and are implemented as LLM-backed agents.
- The mapping above will guide the refactoring in Phase 3 and 4.

---

**Helpful notes for next phases:**
- All tools and agents are now clearly inventoried and classified.
- Proceed to Phase 3: Refactor all tools as stateless, LLM-backed functions compatible with langgraph-supervisor.
- Use the table above as a reference for which components to refactor as tools vs persona agents.
- Ensure tests are updated or created for each tool and agent as you refactor.

---

## **Phase 3: Refactor Tools as LLM-Backed Functions** ✅ *Completed*

**Goal:** Implement all tools as stateless, LLM-backed functions compatible with langgraph-supervisor.

**Status:**  
✅ All tool agents (roll, search, todo, report, knowledge, storyboard_editor, persona_classifier) have been refactored as stateless, async LLM-backed functions using the langgraph-supervisor tool interface.  
✅ All persona or state management has been removed from tools.  
✅ All tools now only process their input and return output.  
✅ Tests for tools are in place and pass.

**Helpful notes for next phases:**  
- All tools are now stateless and ready to be called by persona agents or the supervisor.
- Proceed to Phase 4: Refactor persona agents as LLM-backed agents using persona-specific prompts.
- Ensure that agents use the new stateless tools via the supervisor handoff mechanism.
- Continue to keep tests updated as you refactor agents.

---

## **Phase 4: Refactor Persona Agents** ✅ *Completed*

**Goal:** Implement each persona agent as an LLM-backed agent using langgraph-supervisor.

**Status:**  
✅ All persona agents (Storyteller GM, Therapist, Secretary, Coder, Friend, Lorekeeper, Dungeon Master, Default) are now implemented as stateless, LLM-backed async functions or callable classes using persona-specific Jinja2 prompts.
✅ Each agent uses the correct prompt key from config and is compatible with the langgraph-supervisor orchestration.
✅ Persona agents can call tools via the supervisor’s handoff mechanism.
✅ No custom workflow logic remains; all persona logic is prompt-driven.
✅ Tests for persona agent prompt selection and invocation are present and pass.

**Helpful notes for next phases:**  
- Persona agents are now stateless and prompt-driven, ready for supervisor orchestration.
- The `writer_agent` exposes a registry of persona agents for supervisor handoff.
- Proceed to Phase 5: Implement the supervisor agent to orchestrate persona agents and tools.
- Ensure the supervisor manages message history, persona switching, and tool handoff as per the langgraph-supervisor pattern.
- Continue to keep tests updated as you refactor the supervisor and workflow.

---

## **Phase 5: Implement Supervisor Workflow** ✅ *Completed*

**Goal:** Replace custom Oracle/Director logic with a langgraph-supervisor supervisor agent.

**Status:**  
✅ Supervisor agent implemented in `src/supervisor.py` using the langgraph-supervisor pattern.  
✅ All user input is now routed through the supervisor, which decides which agent or tool to invoke.  
✅ Message history, persona switching, and tool handoff are managed as per langgraph-supervisor conventions.  
✅ `src/workflows.py` updated to use the supervisor as the main workflow entrypoint.  
✅ All tests pass with the new workflow.

**Helpful notes for next phases:**  
- The supervisor agent is now the central orchestrator for all persona agents and tools.
- All routing logic is consolidated in `src/supervisor.py`, making it easy to extend or modify agent/tool selection.
- Proceed to Phase 6: Remove any obsolete code, update documentation, and ensure all tests pass.
- Update CLI and Chainlit event handlers to use the new supervisor workflow if not already done.
- Run `make test` to verify stability after any further changes.

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

## **Useful langgraph-supervisor Patterns and Snippets**

### **Supervisor/Agent/Tool Orchestration**

- **Supervisor receives user input and state, decides which agent/tool to call.**
- **Agents and tools are async functions or classes that take in state and return messages.**
- **Handoff tools can be used to pass control between agents/tools.**

### **Example: Creating a Supervisor with Agents and Tools**

```python
# Define tools
async def roll_tool(state, **kwargs):
    # ... (see Phase 3 tips above)
    pass

async def search_tool(state, **kwargs):
    # ... (see Phase 3 tips above)
    pass

# Define persona agents
async def storyteller_agent(state, **kwargs):
    # ... (see Phase 4 tips above)
    pass

# Supervisor logic
async def supervisor(state, **kwargs):
    user_input = state.get_last_human_message().content
    if "roll" in user_input:
        return await roll_tool(state)
    elif "search" in user_input:
        return await search_tool(state)
    else:
        return await storyteller_agent(state)
```

### **Message History Management**

- **Full history:** Pass all messages to the next agent/tool.
- **Last message only:** Pass only the most recent message.
- **Custom:** Pass a slice or filtered history as needed.

### **Multi-level Hierarchies**

- **You can create supervisors that manage other supervisors or teams of agents.**
- **Each supervisor can be compiled as a workflow and used as a sub-agent.**

### **Customizing Handoff Tools**

- **You can create custom handoff tools to control how tasks are delegated between agents.**
- **Handoff tools can include extra arguments, such as a task description or context.**

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

---

**Proceed phase by phase. After each phase, verify tests and code health before moving to the next.**
