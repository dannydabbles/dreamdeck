import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from src.cli import main

def test_cli_run_agent():
    """Test CLI agent execution"""
    runner = CliRunner()
    with patch("src.cli.run_async") as mock_run:
        mock_run.return_value = [MagicMock(content="Test response", name="Agent")]
        
        result = runner.invoke(main, [
            "run-agent", "writer", 
            "--input", "Test story", 
            "--persona", "Storyteller GM"
        ])
        
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
        
        result = runner.invoke(main, [
            "run-workflow",
            "--input", "Attack the dragon",
            "--persona", "Dungeon Master"
        ])
        
        assert "Dice rolled" in result.output
        assert "Battle ensues" in result.output
