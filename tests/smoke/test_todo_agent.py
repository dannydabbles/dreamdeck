import datetime as dt_module
import os
import shutil
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

# --- Change Start 1: Import config ---
from src.config import config as app_config # Import the actual config or a mockable reference
# --- Change End 1 ---

from src.agents.todo_agent import _manage_todo
from src.models import ChatState


@pytest.mark.asyncio
async def test_manage_todo_creates_file(tmp_path):
    fixed_now = dt_module.datetime.now()
    manager_persona_for_test = "Secretary" # Define the manager persona for this test

    # --- Change Start 2: Add patch for config ---
    with patch("src.agents.todo_agent.datetime") as mock_datetime, patch(
        "src.agents.todo_agent.TODO_DIR_PATH", str(tmp_path)
    ), patch("src.agents.todo_agent.TODO_FILE_NAME", "todo.md"), patch(
        "src.agents.todo_agent.CLMessage", new_callable=MagicMock
    ) as mock_cl_msg_cls, patch(
        "src.agents.todo_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock
    ) as mock_ainvoke, patch(
        "src.agents.todo_agent.cl", new_callable=MagicMock
    ) as mock_cl_module, patch( # Add this patch context manager
        "src.agents.todo_agent.config", new_callable=MagicMock
    ) as mock_config: # Mock the config object used by the agent
    # --- Change End 2 ---

        # Configure the mocked config object
        mock_config.defaults.todo_manager_persona = manager_persona_for_test

        # Patch datetime.utcnow() to fixed_now
        mock_datetime.datetime.utcnow.return_value = fixed_now
        mock_datetime.datetime.strftime = dt_module.datetime.strftime
        mock_datetime.datetime.now.return_value = fixed_now
        # Patch .strftime directly on the datetime object, not on the method
        mock_datetime.datetime.now.return_value.strftime = lambda fmt: fixed_now.strftime(fmt)


        # Patch cl.user_session.get to avoid "Chainlit context not found"
        mock_cl_module.user_session.get.return_value = {}

        mock_cl_msg_instance = AsyncMock()
        mock_cl_msg_instance.send = AsyncMock(return_value=None)
        mock_cl_msg_instance.id = "todo-msg-id"
        mock_cl_msg_cls.return_value = mock_cl_msg_instance

        # Mock LLM response to output a JSON list with the task
        mock_response = MagicMock()
        mock_response.content = '["buy milk"]'
        mock_ainvoke.return_value = mock_response

        state = ChatState(
            thread_id="test",
            messages=[HumanMessage(content="/todo buy milk", name="Player")],
        )
        # Note: state.current_persona is no longer directly used for the file path
        state.current_persona = "Friend" # Set a different persona to ensure it's ignored

        result = await _manage_todo(state)
        assert result
        assert "buy milk" in result[0].content

        current_date = fixed_now.strftime("%Y-%m-%d")
        # --- Change Start 3: Update assertion path ---
        # Assert the file exists under the MANAGER persona's directory
        todo_file = tmp_path / manager_persona_for_test / current_date / "todo.md"
        # --- Change End 3 ---
        assert todo_file.exists()
        content = todo_file.read_text()
        assert "buy milk" in content


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
