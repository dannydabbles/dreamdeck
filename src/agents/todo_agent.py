from src.config import config, cl_logger
import os
import datetime
from chainlit import Message as CLMessage
from langgraph.func import task
from langchain_core.messages import AIMessage
from src.models import ChatState

# Module-level constants for easier patching in tests
TODO_DIR_PATH = config.TODO_DIR_PATH
TODO_FILE_NAME = config.TODO_FILE_NAME

async def _manage_todo(state: ChatState) -> list[AIMessage]:
    try:
        last_human = state.get_last_human_message()
        if not last_human:
            raise ValueError("No user input found.")
        user_input = last_human.content

        if not user_input.startswith("/todo"):
            return []

        task_text = user_input[6:].strip()
        if not task_text:
            raise ValueError("Task cannot be empty")

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        dir_path = os.path.join(
            TODO_DIR_PATH,
            f"helper_for_{current_date}"
        )
        file_path = os.path.join(dir_path, TODO_FILE_NAME)

        os.makedirs(dir_path, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M]")
        with open(file_path, "a") as f:
            f.write(f"\n{timestamp} {task_text}")

        cl_msg = CLMessage(
            content=f"✅ Added: {task_text}",
            parent_id=None
        )
        await cl_msg.send()
        cl_msg_id = cl_msg.id

        return [
            AIMessage(
                content=f"Added: {task_text}",
                name="todo",
                metadata={"message_id": cl_msg_id, "parent_id": None},
            )
        ]

    except Exception as e:
        cl_logger.error(f"Todo failed: {str(e)}")
        return [AIMessage(content=str(e), name="error")]

@task
async def manage_todo(state: ChatState) -> list[AIMessage]:
    return await _manage_todo(state)

todo_agent = manage_todo
