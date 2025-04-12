"""
Central registry for all agents and tools in Dreamdeck.
This enables CLI access, dynamic routing, and auto-generated help.
"""

from src.agents.writer_agent import writer_agent
from src.agents.dice_agent import dice_agent
from src.agents.web_search_agent import web_search_agent
from src.agents.todo_agent import todo_agent
from src.agents.knowledge_agent import knowledge_agent
from src.agents.report_agent import report_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.persona_classifier_agent import persona_classifier_agent
from src.agents.decision_agent import decision_agent

AGENT_REGISTRY = {
    "writer": {
        "agent": writer_agent,
        "description": "Storyteller/GM, generates narrative.",
    },
    "dice": {
        "agent": dice_agent,
        "description": "Dice rolling agent.",
    },
    "search": {
        "agent": web_search_agent,
        "description": "Web search agent.",
    },
    "todo": {
        "agent": todo_agent,
        "description": "TODO list manager.",
    },
    "knowledge": {
        "agent": knowledge_agent,
        "description": "Knowledge/lore agent.",
    },
    "report": {
        "agent": report_agent,
        "description": "Daily report/summary agent.",
    },
    "storyboard": {
        "agent": storyboard_editor_agent,
        "description": "Storyboard image generator.",
    },
    "persona_classifier": {
        "agent": persona_classifier_agent,
        "description": "Suggests best persona for next turn.",
    },
    "decision": {
        "agent": decision_agent,
        "description": "LLM-based agent router (oracle/decision agent).",
    },
    # Add more agents/tools here as needed
}

def get_agent(name):
    """Get agent callable by name (case-insensitive)."""
    if not isinstance(name, str):
        return None
    return AGENT_REGISTRY.get(name.lower(), {}).get("agent")

def list_agents():
    """List all registered agents/tools."""
    return [(k, v["description"]) for k, v in AGENT_REGISTRY.items()]

def describe_agent(name):
    """Return the docstring or description for an agent/tool."""
    agent = get_agent(name)
    if agent is None:
        return None
    doc = getattr(agent, "__doc__", None)
    if doc:
        return doc
    return AGENT_REGISTRY.get(name.lower(), {}).get("description", "")
