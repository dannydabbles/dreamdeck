"""
Supervisor agent for Dreamdeck, orchestrating persona agents and tools
using the langgraph-supervisor pattern, now using the langgraph-supervisor graph API.
"""

from src.models import ChatState
from src.agents.registry import AGENT_REGISTRY, get_agent
from src.agents.writer_agent import writer_agent
from langchain_core.messages import HumanMessage, AIMessage
import logging
import sys
import os

try:
    from langgraph_supervisor.graph import SupervisorGraph
    from langgraph_supervisor.nodes import SupervisorNode, ToolNode, PersonaNode
except ImportError as e:
    raise ImportError(
        "Could not import SupervisorGraph, SupervisorNode, ToolNode, PersonaNode from langgraph_supervisor. "
        "Please ensure you have the latest version of langgraph-supervisor installed. "
        "Try: pip install --upgrade langgraph-supervisor"
    ) from e

cl_logger = logging.getLogger("chainlit")

def _normalize_persona(persona: str) -> str:
    if not persona:
        return "default"
    return persona.lower().replace(" ", "_")

# Define PersonaNode for each persona in writer_agent (if registry exists)
persona_nodes = {}
if hasattr(writer_agent, "persona_agent_registry"):
    for persona_key, agent in writer_agent.persona_agent_registry.items():
        persona_nodes[persona_key] = PersonaNode(
            name=persona_key,
            agent=agent,
        )
else:
    # Fallback: just use writer_agent as default persona
    persona_nodes["default"] = PersonaNode(
        name="default",
        agent=writer_agent,
    )

# Define ToolNodes for each tool in AGENT_REGISTRY
tool_nodes = {}
for tool_name, entry in AGENT_REGISTRY.items():
    tool_nodes[tool_name] = ToolNode(
        name=tool_name,
        agent=entry["agent"],
    )

# SupervisorNode: decision logic for routing
class DreamdeckSupervisorNode(SupervisorNode):
    def __init__(self):
        super().__init__(name="dreamdeck_supervisor")

    async def route(self, state: ChatState, **kwargs):
        last_human = state.get_last_human_message()
        if not last_human:
            cl_logger.warning("Supervisor: No user message found in state.")
            return {"route": "default"}

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

# Build the SupervisorGraph
supervisor_graph = SupervisorGraph(
    supervisor_node=DreamdeckSupervisorNode(),
    persona_nodes=persona_nodes,
    tool_nodes=tool_nodes,
)

# The main entrypoint for Chainlit and tests
async def supervisor(state: ChatState, **kwargs):
    """
    Entrypoint for the Dreamdeck supervisor using langgraph-supervisor.
    """
    # The SupervisorGraph handles routing and execution
    return await supervisor_graph.ainvoke(state, **kwargs)

# Patch: add .ainvoke for test compatibility (LangGraph expects this in tests)
supervisor.ainvoke = supervisor_graph.ainvoke
