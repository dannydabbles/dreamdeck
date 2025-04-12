"""
Supervisor agent for Dreamdeck, orchestrating persona agents and tools
using the langgraph-supervisor pattern, now using the langgraph-supervisor graph API.
"""

from src.models import ChatState
from src.agents.registry import AGENT_REGISTRY, get_agent
from src.agents.writer_agent import writer_agent
from langchain_core.messages import HumanMessage, AIMessage
import logging

from langgraph_supervisor import create_supervisor

cl_logger = logging.getLogger("chainlit")

def _normalize_persona(persona: str) -> str:
    if not persona:
        return "default"
    return persona.lower().replace(" ", "_")

# Custom supervisor agent logic (preserves your routing logic)
async def dreamdeck_supervisor_agent(state: ChatState, **kwargs):
    last_human = state.get_last_human_message()
    if not last_human:
        cl_logger.warning("Supervisor: No user message found in state.")
        return {"route": "default"}
dreamdeck_supervisor_agent.name = "dreamdeck_supervisor"

    user_input = last_human.content.strip().lower()

    # Tool routing: check for explicit tool commands
    for tool_name in AGENT_REGISTRY:
        if user_input.startswith(f"/{tool_name}") or tool_name in user_input:
            cl_logger.info(f"Supervisor: Routing to {tool_name}_agent.")
            if tool_name == "storyboard":
                # Find last GM message id for storyboard
                gm_msg = next(
                    (msg for msg in reversed(state.messages)
                     if isinstance(msg, AIMessage) and msg.name and "game master" in msg.name.lower()
                     and msg.metadata and "message_id" in msg.metadata),
                    None,
                )
                if gm_msg:
                    return {"route": tool_name, "gm_message_id": gm_msg.metadata["message_id"]}
                else:
                    cl_logger.warning("Supervisor: No GM message found for storyboard.")
                    return {"route": "default"}
            return {"route": tool_name}

    # --- LLM-based dynamic routing: call decision agent ---
    decision_agent = get_agent("decision")
    if decision_agent is not None:
        decision = await decision_agent(state)
        route = decision.get("route", "writer")
        cl_logger.info(f"Supervisor: Decision agent route: {route}")
        if route.startswith("persona:"):
            persona_name = route.split(":", 1)[1].strip()
            persona_key = _normalize_persona(persona_name)
            state.current_persona = persona_key.replace("_", " ").title()
            return {"route": f"persona:{persona_key}"}
        elif route in AGENT_REGISTRY:
            if route == "storyboard":
                gm_msg = next(
                    (msg for msg in reversed(state.messages)
                     if isinstance(msg, AIMessage) and msg.name and "game master" in msg.name.lower()
                     and msg.metadata and "message_id" in msg.metadata),
                    None,
                )
                if gm_msg:
                    return {"route": route, "gm_message_id": gm_msg.metadata["message_id"]}
                else:
                    cl_logger.warning("Supervisor: No GM message found for storyboard.")
                    return {"route": "default"}
            return {"route": route}
        else:
            # fallback: treat as persona name
            persona_key = _normalize_persona(route)
            state.current_persona = persona_key.replace("_", " ").title()
            return {"route": f"persona:{persona_key}"}

    # Fallback: route to current persona agent
    persona = getattr(state, "current_persona", "default")
    persona_key = _normalize_persona(persona)
    state.current_persona = persona_key.replace("_", " ").title()
    return {"route": f"persona:{persona_key}"}

# Persona agents
if hasattr(writer_agent, "persona_agent_registry"):
    persona_agents = list(writer_agent.persona_agent_registry.values())
else:
    persona_agents = [writer_agent]

# Tool agents
tool_agents = [entry["agent"] for tool, entry in AGENT_REGISTRY.items()]

# Choose the LLM model for the supervisor (use writer_agent.llm if available, else fallback)
try:
    model = writer_agent.llm
except AttributeError:
    # Fallback: import and instantiate your model here if needed
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model="gpt-4o")

# Create the supervisor workflow using the public API
supervisor_workflow = create_supervisor(
    [dreamdeck_supervisor_agent] + persona_agents + tool_agents,
    model=model,
).compile()

# The main entrypoint for Chainlit and tests
async def supervisor(state: ChatState, **kwargs):
    """
    Entrypoint for the Dreamdeck supervisor using langgraph-supervisor.
    """
    return await supervisor_workflow.ainvoke(state, **kwargs)

# Patch: add .ainvoke for test compatibility (LangGraph expects this in tests)
supervisor.ainvoke = supervisor_workflow.ainvoke
