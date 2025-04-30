import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
import importlib

def test_cli_run_agent():
    """Test CLI agent execution"""
    runner = CliRunner()
    with patch("src.cli.run_async") as mock_run:
        mock_run.return_value = [MagicMock(content="Test response", name="Agent")]
        cli_mod = importlib.import_module("src.cli")
        # Patch .name attribute to simulate Click command object
        cli_mod.main.name = "main"
        result = runner.invoke(cli_mod.main, [
            "run-agent", "writer", 
            "--input", "Test story", 
            "--persona", "Storyteller GM"
        ])
        # Defensive: print result.exception for debugging if output is empty
        if not result.output:
            print("CLI test_cli_run_agent result.exception:", result.exception)
        assert "Test response" in result.output
        assert "Agent" in result.output

def test_cli_workflow_execution():
    """Test full workflow execution via CLI"""
    runner = CliRunner()
    with patch("src.cli.run_async") as mock_run:
        mock_run.return_value = [
            MagicMock(content="Dice rolled: 15", name="dice_roll"),
            MagicMock(content="Battle ensues!", name="Storyteller GM")
        ]
        cli_mod = importlib.import_module("src.cli")
        # Patch .name attribute to simulate Click command object
        cli_mod.main.name = "main"
        result = runner.invoke(cli_mod.main, [
            "run-workflow",
            "--input", "Attack the dragon",
            "--persona", "Dungeon Master"
        ])
        if not result.output:
            print("CLI test_cli_workflow_execution result.exception:", result.exception)
        assert "Dice rolled" in result.output
        assert "Battle ensues" in result.output
