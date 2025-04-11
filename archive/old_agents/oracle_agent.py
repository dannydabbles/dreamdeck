import json
import logging
from jinja2 import Template
from langgraph.func import task
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage

from src.config import config, LLM_TIMEOUT, OPENAI_SETTINGS
from src.models import ChatState
from src.agents import agents_map  # Import agents_map to get available agents

import chainlit as cl

cl_logger = logging.getLogger("chainlit")


@cl.step(name="Oracle Agent: Decide Next Action", type="tool")
async def _oracle_decision(state: ChatState, **kwargs) -> str:
    """
    The Oracle decides the next single action based on the conversation state.

    Returns:
        str: The name of the agent to call next, or "END_TURN".
    """
    try:
        # Defensive: fallback to a default prompt if missing
        prompt_template_str = config.loaded_prompts.get(
            "oracle_decision_prompt",
            "<goal>You are the Oracle. Output a JSON: {\"next_action\": \"END_TURN\"}</goal>"
        )
        template = Template(prompt_template_str)

        # Get available agent names from the map
        available_agents = list(agents_map.keys())

        # Phase 3 Placeholder: Add tool_results_this_turn later
        tool_results_this_turn_str = ""
        if hasattr(state, "tool_results_this_turn") and state.tool_results_this_turn:
             tool_results_this_turn_str = "\n".join(
                 [f"{msg.name}: {msg.content}" for msg in state.tool_results_this_turn]
             )

        formatted_prompt = template.render(
            current_persona=state.current_persona,
            available_agents=available_agents,
            recent_chat_history=state.get_recent_history_str(),
            tool_results_this_turn=tool_results_this_turn_str, # Phase 3 Placeholder
            max_iterations=config.max_chain_length,
        )

        # Use generic LLM settings for now, can be customized later
        llm = ChatOpenAI(
            base_url=OPENAI_SETTINGS.get("base_url"),
            temperature=0.1, # Low temp for deterministic decision
            max_tokens=150,
            streaming=False,
            verbose=True,
            timeout=LLM_TIMEOUT,
        )

        response = await llm.ainvoke([("system", formatted_prompt)])
        content = response.content.strip()
        cl_logger.info(f"Oracle raw response: {content}")

        try:
            # Remove markdown code fencing if present
            if content.startswith("```") and content.endswith("```"):
                lines = content.splitlines()
                if len(lines) >= 3:
                    # Handle potential json language tag
                    if lines[0].strip().lower() == "```json":
                         content = "\n".join(lines[1:-1]).strip()
                    else:
                         content = "\n".join(lines[1:-1]).strip() # Assume no language tag if not json

            parsed = json.loads(content)
            next_action = parsed.get("next_action", "END_TURN").strip()

            # Validate action
            if next_action != "END_TURN" and next_action not in available_agents:
                cl_logger.warning(
                    f"Oracle suggested unknown action '{next_action}'. Defaulting to END_TURN."
                )
                return "END_TURN"

            cl_logger.info(f"Oracle decided next action: {next_action}")
            return next_action

        except json.JSONDecodeError as e:
            cl_logger.error(
                f"Failed to parse Oracle JSON response: {content}. Error: {e}. Defaulting to END_TURN."
            )
            return "END_TURN"
        except Exception as e:
             cl_logger.error(
                f"Error processing Oracle response: {content}. Error: {e}. Defaulting to END_TURN."
            )
             return "END_TURN"

    except Exception as e:
        cl_logger.error(f"Oracle agent failed: {e}", exc_info=True)
        return "END_TURN" # Default to ending turn on any failure


@task
async def oracle_agent(state: ChatState, **kwargs) -> str:
    """LangGraph task wrapper for the Oracle decision agent."""
    return await _oracle_decision(state, **kwargs)
