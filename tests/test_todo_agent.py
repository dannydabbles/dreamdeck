import pytest
import os
import shutil
from unittest.mock import patch, AsyncMock
from unittest.mock import MagicMock
from src.agents.todo_agent import _manage_todo
from src.models import ChatState
from langchain_core.messages import HumanMessage

@pytest.mark.asyncio
async def test_manage_todo_creates_file(tmp_path):
    # Patch module-level constants, not the config object
    with patch("src.agents.todo_agent.TODO_DIR_PATH", str(tmp_path)), \
         patch("src.agents.todo_agent.TODO_FILE_NAME", "todo.md"), \
         patch("src.agents.todo_agent.CLMessage", new_callable=MagicMock) as mock_cl_msg_cls:

        mock_cl_msg_instance = MagicMock()
        mock_cl_msg_instance.send = AsyncMock()
        mock_cl_msg_instance.send.return_value = None
        mock_cl_msg_instance.id = "todo-msg-id"
        mock_cl_msg_cls.return_value = mock_cl_msg_instance

        state = ChatState(
            thread_id="test",
            messages=[HumanMessage(content="/todo buy milk", name="Player")]
        )

        result = await _manage_todo(state)
        # Check AIMessage returned
        assert result
        assert "Added" in result[0].content

        # Check file created
        date_dir = next(tmp_path.iterdir())
        todo_file = date_dir / "todo.md"
        assert todo_file.exists()
        content = todo_file.read_text()
        assert "buy milk" in content

@pytest.mark.asyncio
async def test_manage_todo_empty_task():
    state = ChatState(
        thread_id="test",
        messages=[HumanMessage(content="/todo ", name="Player")]
    )
    result = await _manage_todo(state)
    assert "Task cannot be empty" in result[0].content

@pytest.mark.asyncio
async def test_manage_todo_no_human():
    state = ChatState(thread_id="test", messages=[])
    result = await _manage_todo(state)
    assert "No user input" in result[0].content
