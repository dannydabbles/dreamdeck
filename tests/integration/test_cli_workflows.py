from unittest.mock import patch, MagicMock
import sys
import pytest
from src.cli import main  # Add this import


def test_cli_run_agent():
    """Test CLI agent execution"""
    test_args = [
        "prog",
        "run-agent",
        "writer",
        "--input",
        "Test story",
        "--persona",
        "Storyteller GM",
    ]

    with patch.object(sys, "argv", test_args), patch("src.cli.run_async") as mock_run:

        mock_run.return_value = [MagicMock(content="Test response", name="Agent")]

        # Capture stdout
        from io import StringIO

        sys.stdout = StringIO()

        main()  # Call main directly

        output = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__

        assert "Test response" in output
        assert "Agent" in output


def test_cli_workflow_execution():
    """Test full workflow execution via CLI"""
    test_args = [
        "prog",
        "run-workflow",
        "--input",
        "Attack the dragon",
        "--persona",
        "Dungeon Master",
    ]

    with patch.object(sys, "argv", test_args), patch("src.cli.run_async") as mock_run:

        mock_run.return_value = [
            MagicMock(content="Dice rolled: 15", name="dice_roll"),
            MagicMock(content="Battle ensues!", name="Storyteller GM"),
        ]

        # Capture stdout
        from io import StringIO

        sys.stdout = StringIO()

        main()  # Call main directly

        output = sys.stdout.getvalue()
        sys.stdout = sys.__stdout__

        assert "Dice rolled" in output
        assert "Battle ensues" in output
