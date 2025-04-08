from src.config import config, cl_logger
import os
import datetime
from chainlit import Message as CLMessage
from langgraph.func import task
from langchain_core.messages import AIMessage
from src.models import ChatState

# Module-level constants for easier patching in tests
TODO_DIR_PATH = config.todo_dir_path
TODO_FILE_NAME = config.todo_file_name

async def _manage_todo(state: ChatState) -> list[AIMessage]:
    try:
        cl_logger.info("Starting _manage_todo()")

        last_human = state.get_last_human_message()
        cl_logger.info(f"Last human message: {last_human}")

        if not last_human:
            raise ValueError("No user input found.")
        user_input = last_human.content
        cl_logger.info(f"User input: {user_input}")

        if not user_input.startswith("/todo"):
            return []

        task_text = user_input[6:].strip()
        cl_logger.info(f"Task text: '{task_text}'")

        if not task_text:
            raise ValueError("Task cannot be empty")

        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        dir_path = TODO_DIR_PATH
        file_path = os.path.join(dir_path, TODO_FILE_NAME)

        cl_logger.info(f"Todo directory path: {dir_path}")
        cl_logger.info(f"Todo file path: {file_path}")

        os.makedirs(dir_path, exist_ok=True)
        cl_logger.info(f"Ensured directory exists: {dir_path}")

        timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M]")
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(f"\n{timestamp} {task_text}")

        cl_logger.info(f"Wrote task to file: {file_path}")

        cl_msg = CLMessage(
            content=f"✅ Added to TODO:\n{task_text}",
            parent_id=None
        )
        await cl_msg.send()
        cl_msg_id = cl_msg.id

        cl_logger.info(f"Sent Chainlit message with id: {cl_msg_id}")

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
