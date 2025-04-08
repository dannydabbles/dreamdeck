from src.config import config, cl_logger
import os
import datetime
from chainlit import Message as CLMessage
import chainlit as cl
from langgraph.func import task
from langchain_core.messages import AIMessage
from src.models import ChatState
from langchain_openai import ChatOpenAI
from jinja2 import Template

# Module-level constants for easier patching in tests
TODO_DIR_PATH = config.todo_dir_path
TODO_FILE_NAME = config.todo_file_name


async def _manage_todo(state: ChatState) -> list[AIMessage]:
    try:
        last_human = state.get_last_human_message()
        if not last_human:
            return [AIMessage(content="No user input found.", name="error")]
        user_input = last_human.content.strip()

        # Special case: if user_input is empty or just "/todo", treat as empty task
        if user_input.startswith("/todo"):
            task_text = user_input[5:].strip()
            if not task_text:
                return [AIMessage(content="Task cannot be empty", name="error")]
            user_input = task_text

        # Load current todo list from file (if exists)
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        dir_path = os.path.join(TODO_DIR_PATH, current_date)
        file_path = os.path.join(dir_path, TODO_FILE_NAME)

        existing_content = ""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                existing_content = f.read()

        # Prepare prompt
        template = Template(config.loaded_prompts.get("todo_prompt", ""))
        prompt = template.render(
            existing_todo_file=existing_content,
            user_input=user_input,
            recent_chat_history=state.get_recent_history_str(),
            tool_results=state.get_tool_results_str(),
        )

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

        # Save updated markdown back to file
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_markdown)

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
        return [AIMessage(content="Todo update failed.", name="error")]


@task
async def manage_todo(state: ChatState) -> list[AIMessage]:
    return await _manage_todo(state)


todo_agent = manage_todo


async def call_todo_agent(state: ChatState) -> list[AIMessage]:
    return await _manage_todo(state)
