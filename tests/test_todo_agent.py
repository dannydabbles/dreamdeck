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
         patch("src.agents.todo_agent.CLMessage", new_callable=MagicMock) as mock_cl_msg_cls, \
         patch("src.agents.todo_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke, \
         patch("src.agents.todo_agent.cl", new_callable=MagicMock) as mock_cl_module:

        # Patch cl.user_session.get to avoid "Chainlit context not found"
        mock_cl_module.user_session.get.return_value = {}

        mock_cl_msg_instance = MagicMock()
        mock_cl_msg_instance.send = AsyncMock()
        mock_cl_msg_instance.send.return_value = None
        mock_cl_msg_instance.id = "todo-msg-id"
        mock_cl_msg_cls.return_value = mock_cl_msg_instance

        # Mock LLM response to output a JSON list with the task
        mock_response = MagicMock()
        mock_response.content = '["buy milk"]'
        mock_ainvoke.return_value = mock_response

        state = ChatState(
            thread_id="test",
            messages=[HumanMessage(content="/todo buy milk", name="Player")]
        )

        result = await _manage_todo(state)
        # Check AIMessage returned
        assert result
        # The LLM output is now a markdown with sections, so check for the task inside Remaining or In Progress
        assert "buy milk" in result[0].content
        assert "## Finished" in result[0].content
        assert "## In Progress" in result[0].content
        assert "## Remaining" in result[0].content

        # Check file created inside date-based subfolder
        import datetime
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")
        todo_file = tmp_path / current_date / "todo.md"
        assert todo_file.exists()
        content = todo_file.read_text()
        assert "buy milk" in content
        assert "## Finished" in content
        assert "## In Progress" in content
        assert "## Remaining" in content

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
