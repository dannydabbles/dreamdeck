from src.config import config, OPENAI_SETTINGS, LLM_TIMEOUT
import logging
from jinja2 import Template
from langgraph.func import task
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from src.models import ChatState
import chainlit as cl

cl_logger = logging.getLogger("chainlit")

# Define the fixed list of personas
PERSONA_LIST = [
    "storyteller_gm",
    "therapist",
    "secretary",
    "coder",
    "friend",
    "lorekeeper",
    "dungeon_master",
    "default",
]

# Create a simple prompt template string
CLASSIFIER_PROMPT = """
You are an AI persona classifier.

Given the recent chat history, memories, and tool results, suggest the most appropriate persona to handle the next user input.

Choose ONLY from this list:
{{ persona_list }}

Output a JSON object with two keys:
- "persona": the suggested persona string (must be one of the above)
- "reason": a brief explanation (max 1 sentence)

If unsure, default to "default".

Recent chat:
{{ recent_chat_history }}

Memories:
{{ memories }}

Tool results:
{{ tool_results }}

Respond ONLY with the JSON object, no extra text.
"""

@cl.step(name="Persona Classifier Agent", type="tool")
async def _classify_persona(state: ChatState) -> dict:
    try:
        template = Template(CLASSIFIER_PROMPT)
        prompt = template.render(
            persona_list=", ".join(PERSONA_LIST),
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
            tool_results=state.get_tool_results_str(),
        )

        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("persona_classifier_temp", 0.2)
        final_endpoint = user_settings.get("persona_classifier_endpoint") or OPENAI_SETTINGS.get("base_url")
        final_max_tokens = user_settings.get("persona_classifier_max_tokens", 200)

        llm = ChatOpenAI(
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=False,
            verbose=True,
            timeout=LLM_TIMEOUT,
        )

        response = await llm.ainvoke([("system", prompt)])
        content = response.content.strip()

        # Remove markdown code fencing if present
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            if len(lines) >= 3:
                content = "\n".join(lines[1:-1]).strip()

        import json
        parsed = json.loads(content)
        persona = parsed.get("persona", "default").strip().lower()
        reason = parsed.get("reason", "")

        if persona not in PERSONA_LIST:
            cl_logger.warning(f"Classifier suggested unknown persona '{persona}', defaulting to 'default'")
            persona = "default"

        cl_logger.info(f"Persona classifier suggests: {persona} (reason: {reason})")

        # Save suggestion in Chainlit user session
        cl.user_session.set("suggested_persona", {"persona": persona, "reason": reason})

        return {"persona": persona, "reason": reason}

    except Exception as e:
        cl_logger.error(f"Persona classifier failed: {e}")
        # On failure, fallback to current persona or default
        cl.user_session.set("suggested_persona", {"persona": "default", "reason": "Classifier error"})
        return {"persona": "default", "reason": "Classifier error"}

@task
async def persona_classifier_agent(state: ChatState, **kwargs) -> dict:
    return await _classify_persona(state)
