from src.config import config, cl_logger
import os
import datetime
import re
from chainlit import Message as CLMessage
import chainlit as cl
from langgraph.func import task
from langchain_core.messages import AIMessage, HumanMessage
from src.models import ChatState
from langchain_openai import ChatOpenAI
from jinja2 import Template

# Module-level constants for easier patching in tests
TODO_DIR_PATH = config.todo_dir_path
TODO_FILE_NAME = config.todo_file_name

# Exported agent alias for patching
class _TodoAgentWrapper:
    pass

todo_agent = _TodoAgentWrapper()


@cl.step(name="Todo Agent", type="tool")
async def _manage_todo(state: ChatState) -> list[AIMessage]:
    try:
        last_human = state.get_last_human_message()
        if not last_human:
            return [AIMessage(content="No user input found.", name="error", metadata={"message_id": None})]
        original_user_input = last_human.content.strip()

        # Special case: if original_user_input is empty or just "/todo", treat as empty task
        if original_user_input.startswith("/todo"):
            task_text = original_user_input[5:].strip()
            if not task_text:
                return [AIMessage(content="Task cannot be empty", name="error", metadata={"message_id": None})]
        else:
            task_text = original_user_input  # fallback, e.g., if user didn't use slash command

        # Pass the *full* original user input (including slash command) to the prompt for clarity

        # Load current todo list from file (if exists)
        current_date = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        persona = getattr(state, "current_persona", "Default")
        persona_safe = re.sub(r'[^\w\-_. ]', '_', persona)
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
            persona_configs = getattr(config.agents, "todo_agent", {}).get("personas", {})
            if isinstance(persona_configs, dict):
                persona_entry = persona_configs.get(persona)
                if persona_entry and isinstance(persona_entry, dict):
                    prompt_key = persona_entry.get("prompt_key", prompt_key)
        except Exception:
            pass  # fallback to default

        prompt_template_str = config.loaded_prompts.get(prompt_key, "").strip()
        if not prompt_template_str:
            cl_logger.error("Todo prompt template is empty!")
            prompt = f"User input: {original_user_input}\nExisting TODO:\n{existing_content}"
        else:
            template = Template(prompt_template_str)
            prompt = template.render(
                existing_todo_file=existing_content,
                user_input=original_user_input,
                recent_chat_history=state.get_recent_history_str(),
                tool_results=state.get_tool_results_str(),
            )

        cl_logger.info(f"Todo prompt:\n{prompt}")

        # Call LLM
        user_settings = cl.user_session.get("chat_settings", {})
        final_temp = user_settings.get("todo_temp", 0.3)
        final_endpoint = user_settings.get("todo_endpoint") or config.openai.get("base_url")
        final_max_tokens = user_settings.get("todo_max_tokens", 300)

        llm = ChatOpenAI(
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
                    updated_markdown = "**Finished**\n\n**In Progress**\n\n**Remaining**\n" + "\n".join(f"- {item}" for item in items)
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

        cl_msg = CLMessage(
            content=f"📝 Updated TODO list:\n{updated_markdown}",
            parent_id=None,
        )
        await cl_msg.send()

        return [
            AIMessage(
                content=f"Updated TODO list:\n{updated_markdown}",
                name="todo",
                metadata={"message_id": cl_msg.id},
            )
        ]

    except Exception as e:
        cl_logger.error(f"Todo agent failed: {e}")
        return [AIMessage(content="Todo update failed.", name="error", metadata={"message_id": None})]


@task
async def manage_todo(state: ChatState) -> list[AIMessage]:
    return await _manage_todo(state)


# Expose internal function for monkeypatching in tests
_manage_todo = _manage_todo

# Also assign todo_agent._manage_todo for patching in tests
todo_agent._manage_todo = _manage_todo


async def call_todo_agent(state: ChatState, query: str = "") -> list[AIMessage]:
    if query:
        synthetic_msg = HumanMessage(content=f"/todo {query}", name="Player", metadata={"message_id": None})
        state.messages.append(synthetic_msg)
    return await _manage_todo(state)
