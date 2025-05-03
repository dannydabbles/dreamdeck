import datetime as dt_module
import os
import shutil

# Import mock_open from unittest.mock
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest
from langchain_core.messages import HumanMessage

# Import the actual config or a mockable reference
from src.config import config as app_config

from src.agents.todo_agent import _manage_todo
from src.models import ChatState


@pytest.mark.asyncio
async def test_manage_todo_creates_file(tmp_path):
    fixed_date_str = "2024-01-15"  # Use a fixed string for simplicity
    manager_persona_for_test = "Secretary"

    # --- Change Start: Refine patching with os.makedirs and open ---
    # Note: Patching 'open' requires the full path 'src.agents.todo_agent.open'
    with patch("src.agents.todo_agent.datetime") as mock_datetime, patch(
        "src.agents.todo_agent.TODO_DIR_PATH", str(tmp_path)
    ), patch("src.agents.todo_agent.TODO_FILE_NAME", "todo.md"), patch(
        "src.agents.todo_agent.CLMessage", new_callable=MagicMock
    ) as mock_cl_msg_cls, patch(
        "src.agents.todo_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
    ) as mock_ainvoke, patch(
        "src.agents.todo_agent.cl", new_callable=MagicMock
    ) as mock_cl_module, patch(
        "src.agents.todo_agent.config", new_callable=MagicMock
    ) as mock_config, patch(
        "src.agents.todo_agent.os.makedirs"
    ) as mock_makedirs, patch(
        "src.agents.todo_agent.open", mock_open()
    ) as mocked_file, patch(
        "src.agents.todo_agent.os.path.exists", return_value=False
    ) as mock_exists:

        # Patch config.llm.model and config.openai.get("base_url") to valid strings
        mock_config.llm.model = "gpt-4o"
        mock_config.openai.get.return_value = "http://localhost:5000/v1"
        mock_config.defaults.todo_manager_persona = manager_persona_for_test
        mock_config.loaded_prompts.get.return_value = "{{ user_input }}"
        mock_config.todo_dir_path = str(tmp_path)
        mock_config.todo_file_name = "todo.md"

        # Mock the specific strftime call result directly
        mock_datetime.datetime.now.return_value.strftime.return_value = fixed_date_str

        # Mock cl.user_session.get
        mock_cl_module.user_session.get.return_value = {}

        # Mock CLMessage instance
        mock_cl_msg_instance = AsyncMock()
        mock_cl_msg_instance.send = AsyncMock(return_value=None)
        mock_cl_msg_instance.id = "todo-msg-id"
        mock_cl_msg_cls.return_value = mock_cl_msg_instance

        # Mock LLM response (simple JSON for testing conversion)
        mock_response = MagicMock()
        mock_response.content = '["buy milk"]'
        mock_ainvoke.return_value = mock_response

        # Prepare state
        state = ChatState(
            thread_id="test",
            messages=[HumanMessage(content="/todo buy milk", name="Player")],
        )
        state.current_persona = "Friend"  # Ensure this doesn't interfere

        # Run the agent
        result = await _manage_todo(state)
        assert result
        # Check the content returned in the AIMessage (which should be the updated markdown)
        expected_md_content = (
            "**Finished**\n\n**In Progress**\n\n**Remaining**\n- buy milk"
        )
        assert expected_md_content in result[0].content

        # Assertions
        expected_dir = tmp_path / manager_persona_for_test / fixed_date_str
        expected_file = expected_dir / "todo.md"

        # Check os.path.exists was called (for the read part)
        mock_exists.assert_called_once_with(str(expected_file))

        # Check os.makedirs was called correctly (it's called twice now, once before read, once before write)
        # Since exists returns False, the first makedirs isn't called.
        # The second makedirs (before write) IS called.
        mock_makedirs.assert_called_with(str(expected_dir), exist_ok=True)
        # Allow for potential multiple calls if logic changes, check at least one correct call
        assert mock_makedirs.call_count >= 1

        # Check that open was called with the correct file path and mode for writing
        # Since os.path.exists is mocked to return False, the read call to open() is skipped.
        mocked_file.assert_called_once_with(str(expected_file), "w", encoding="utf-8")

        # Check the content written to the file handle
        # mocked_file() gives the mock file handle
        mocked_file().write.assert_called_once_with(expected_md_content)
    # --- Change End ---


@pytest.mark.asyncio
async def test_manage_todo_empty_task():
    state = ChatState(
        thread_id="test", messages=[HumanMessage(content="/todo ", name="Player")]
    )
    result = await _manage_todo(state)
    assert "Task cannot be empty" in result[0].content


@pytest.mark.asyncio
async def test_manage_todo_no_human():
    state = ChatState(thread_id="test", messages=[])
    result = await _manage_todo(state)
    assert "No user input" in result[0].content
