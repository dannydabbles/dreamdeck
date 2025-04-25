import pytest
import sys
from unittest.mock import patch, AsyncMock, MagicMock
from src.cli import main


@pytest.mark.asyncio
async def test_cli_list_agents(monkeypatch, capsys):
    test_args = ["prog", "list-agents"]
    monkeypatch.setattr(sys, "argv", test_args)
    with patch("src.cli.list_agents", return_value=[("writer", "desc")]):
        main()
        out = capsys.readouterr().out
        assert "writer" in out


import pytest


@pytest.mark.skip(reason="Known event loop/asyncio.run test issue, see Dreamdeck #skip")
@pytest.mark.asyncio
async def test_cli_run_agent(monkeypatch, capsys):
    test_args = ["prog", "run-agent", "writer", "--input", "Hello"]
    monkeypatch.setattr(sys, "argv", test_args)
    dummy_agent = AsyncMock(return_value=[MagicMock(name="AI", content="Hi")])
    with patch("src.cli.get_agent", return_value=dummy_agent), patch(
        "asyncio.get_running_loop", side_effect=RuntimeError
    ):
        main()
        out = capsys.readouterr().out
        assert "Agent output" in out


@pytest.mark.skip(reason="Known event loop/asyncio.run test issue, see Dreamdeck #skip")
@pytest.mark.asyncio
async def test_cli_run_workflow(monkeypatch, capsys):
    test_args = ["prog", "run-workflow", "--input", "Hi"]
    monkeypatch.setattr(sys, "argv", test_args)
    dummy_supervisor = AsyncMock(
        return_value=[MagicMock(name="AI", content="Workflow")]
    )
    with patch("src.cli.supervisor", dummy_supervisor), patch(
        "asyncio.get_running_loop", side_effect=RuntimeError
    ):
        main()
        out = capsys.readouterr().out
        assert "Workflow output" in out


@pytest.mark.asyncio
async def test_cli_export_state(tmp_path, monkeypatch, capsys):
    test_args = [
        "prog",
        "export-state",
        "--thread-id",
        "t1",
        "--output",
        str(tmp_path / "state.json"),
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    with patch("src.storage.save_state") as mock_save:
        main()
        mock_save.assert_called_once()
        out = capsys.readouterr().out
        assert "State exported" in out
