import datetime
import os
import re
import zoneinfo

import chainlit as cl
from chainlit import Message as CLMessage
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.func import task

from src.config import cl_logger, config
from src.models import ChatState

# Module-level constants for easier patching in tests
TODO_DIR_PATH = config.todo_dir_path
TODO_FILE_NAME = config.todo_file_name


# Refactored: todo_agent is now a stateless, LLM-backed function (task)
@task
async def todo_agent(state: ChatState, **kwargs) -> list[AIMessage]:
    return await _manage_todo(state, **kwargs)


# Helper for non-langgraph context (slash commands, CLI, etc)
async def todo_agent_helper(state: ChatState, **kwargs) -> list[AIMessage]:
    return await _manage_todo(state, **kwargs)


@cl.step(name="Todo Agent", type="tool")
async def _manage_todo(state: ChatState, **kwargs) -> list[AIMessage]:
    try:
        last_human = state.get_last_human_message()
        if not last_human:
            return [
                AIMessage(
                    content="No user input found.",
                    name="error",
                    metadata={"message_id": None},
                )
            ]
        original_user_input = last_human.content.strip()

        # Only handle explicit /todo slash command at the start
        if original_user_input.startswith("/todo"):
            task_text = original_user_input[5:].strip()
            if not task_text:
                return [
                    AIMessage(
                        content="Task cannot be empty",
                        name="error",
                        metadata={"message_id": None},
                    )
                ]
            # For slash command, pass only the task text to the prompt
            user_input_for_prompt = task_text
        else:
            # For all other cases, pass the full user input to the prompt
            user_input_for_prompt = original_user_input

        # Pass the *full* original user input (including slash command) to the prompt for clarity

        # --- Change Start ---
        # Determine the persona directory based on the configured manager, not the current state persona
        manager_persona = config.defaults.todo_manager_persona
        cl_logger.info(f"Using configured TODO manager persona for file path: {manager_persona}")
        persona_safe = re.sub(r"[^\w\-_. ]", "_", manager_persona)
        # --- Change End ---

        # Load current todo list from file (if exists) using the manager's path
        pacific = zoneinfo.ZoneInfo("America/Los_Angeles")
        current_date = datetime.datetime.now(pacific).strftime("%Y-%m-%d")
        dir_path = os.path.join(TODO_DIR_PATH, persona_safe, current_date)
        file_path = os.path.join(dir_path, TODO_FILE_NAME)

        existing_content = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

        # Prepare prompt
        persona = getattr(state, "current_persona", "Default")
        prompt_key = "todo_prompt"
        try:
            persona_configs = getattr(config.agents, "todo_agent", {}).get(
                "personas", {}
            )
            if isinstance(persona_configs, dict):
                persona_entry = persona_configs.get(persona)
                if persona_entry and isinstance(persona_entry, dict):
                    prompt_key = persona_entry.get("prompt_key", prompt_key)
        except Exception:
            pass  # fallback to default

        prompt_template_str = config.loaded_prompts.get(prompt_key, "").strip()
        if not prompt_template_str:
            cl_logger.error("Todo prompt template is empty!")
            prompt = (
                f"User input: {user_input_for_prompt}\nExisting TODO:\n{existing_content}"
            )
        else:
            template = Template(prompt_template_str)
            prompt = template.render(
                existing_todo_file=existing_content,
                user_input=user_input_for_prompt,
                recent_chat_history=state.get_recent_history_str(),
                tool_results=state.get_tool_results_str(),
            )

        cl_logger.info(f"Todo prompt:\n{prompt}")

        # Call LLM
        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("todo_temp", 0.3)
        final_endpoint = user_settings.get("todo_endpoint") or config.openai.get(
            "base_url"
        )
        final_max_tokens = user_settings.get("todo_max_tokens", 300)

        llm = ChatOpenAI(
            model=config.llm.model,
            base_url=final_endpoint,
            temperature=final_temp,
            max_tokens=final_max_tokens,
            streaming=False,
            verbose=True,
            timeout=config.llm.timeout,
        )

        response = await llm.ainvoke([("system", prompt)])
        updated_markdown = response.content.strip()

        # Defensive: ensure directory exists before saving
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            cl_logger.warning(f"Failed to create todo directory {dir_path}: {e}")

        # Defensive: if markdown looks like a JSON list (e.g., ["buy milk"]), convert to minimal markdown
        if updated_markdown.startswith("[") and updated_markdown.endswith("]"):
            import json

            try:
                items = json.loads(updated_markdown)
                if isinstance(items, list):
                    updated_markdown = (
                        "**Finished**\n\n**In Progress**\n\n**Remaining**\n"
                        + "\n".join(f"- {item}" for item in items)
                    )
            except Exception:
                pass  # fallback to raw content if parse fails

        # Save updated markdown back to file
        try:
            # Defensive: ensure directory exists before saving
            os.makedirs(dir_path, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_markdown)
        except Exception as e:
            cl_logger.warning(f"Failed to write todo file {file_path}: {e}")

        # Generate a unique ID for the message metadata
        import uuid

        message_id = str(uuid.uuid4())

        # Return an AIMessage with fixed name 'todo' for test compatibility
        return [
            AIMessage(
                content=f"Updated TODO list:\n{updated_markdown}",
                name="todo",
                metadata={"message_id": message_id},
            )
        ]

    except Exception as e:
        cl_logger.error(f"Todo agent failed: {e}")
        return [
            AIMessage(
                content="Todo update failed.",
                name="error",
                metadata={"message_id": None},
            )
        ]


@task
async def manage_todo(state: ChatState, **kwargs) -> list[AIMessage]:
    return await _manage_todo(state, **kwargs)


# Expose internal function for monkeypatching in tests
_manage_todo = _manage_todo

# Patch target compatibility: make manage_todo point to undecorated function
manage_todo = _manage_todo

# Also assign todo_agent._manage_todo for patching in tests
todo_agent._manage_todo = _manage_todo


async def call_todo_agent(state: ChatState, query: str = "") -> list[AIMessage]:
    if query:
        synthetic_msg = HumanMessage(
            content=f"/todo {query}", name="Player", metadata={"message_id": None}
        )
        state.messages.append(synthetic_msg)
    return await _manage_todo(state)
