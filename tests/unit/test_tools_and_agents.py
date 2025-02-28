import pytest
from src.tools_and_agents import (
    dice_roll,
    parse_dice_input,
    web_search,
    handle_dice_roll,
)
from unittest.mock import patch, MagicMock
import requests
from typing import List, Tuple


@pytest.fixture
def mock_random():
    with patch("random.randint") as mock_rand:
        mock_rand.return_value = 5
        yield mock_rand


@pytest.fixture
def mock_requests():
    with patch("requests.get") as mock_get:
        yield mock_get


@pytest.fixture
def mock_config():
    with patch.dict(
        "src.tools_and_agents.__dict__",
        {
            "WEB_SEARCH_ENABLED": True,
            "SERPAPI_KEY": "test_key",
            "DICE_ROLLING_ENABLED": True,
        },
    ):
        yield


def test_parse_dice_input_single_die() -> None:
    input_str = "d20"
    result: List[Tuple[int, int]] = parse_dice_input(input_str)
    assert result == [(20, 1)]


def test_parse_dice_input_multiple_dice() -> None:
    input_str = "2d6"
    result: List[Tuple[int, int]] = parse_dice_input(input_str)
    assert result == [(6, 2)]


def test_parse_dice_input_complex() -> None:
    input_str = "3d4 + 1d8"
    result: List[Tuple[int, int]] = parse_dice_input(input_str)
    assert result == [(4, 3), (8, 1)]


def test_parse_dice_input_invalid() -> None:
    input_str = "invalid"
    with pytest.raises(ValueError, match="Invalid dice specification"):
        parse_dice_input(input_str)


def test_parse_dice_input_zero_sides() -> None:
    input_str = "0d6"
    with pytest.raises(ValueError, match="Invalid dice sides"):
        parse_dice_input(input_str)


def test_parse_dice_input_negative_sides() -> None:
    input_str = "-1d6"
    with pytest.raises(ValueError, match="Invalid dice sides"):
        parse_dice_input(input_str)


@pytest.mark.asyncio
async def test_dice_roll_single_die(mock_random: MagicMock, mock_config: None) -> None:
    result: str = await dice_roll()
    assert "🎲 You rolled a 5 on a 20-sided die." in result


@pytest.mark.asyncio
async def test_dice_roll_multiple_dice(mock_random: MagicMock, mock_config: None) -> None:
    result: str = await dice_roll("2d6")
    assert "🎲 You rolled [5, 5] (total: 10) on 2d6." in result


@pytest.mark.asyncio
async def test_dice_roll_invalid_input(mock_config: None) -> None:
    result: str = await dice_roll("invalid")
    assert "🎲 Error rolling dice: Invalid dice specification" in result


@pytest.mark.asyncio
async def test_dice_roll_disabled(mock_config: None) -> None:
    with patch("src.tools_and_agents.DICE_ROLLING_ENABLED", False):
        result: str = await dice_roll()
        assert "Dice rolling is disabled in the configuration." in result


@pytest.mark.asyncio
async def test_dice_roll_zero_sides(mock_random: MagicMock, mock_config: None) -> None:
    with patch("src.tools_and_agents.parse_dice_input", return_value=[(0, 1)]):
        result: str = await dice_roll("0d6")
        assert "🎲 Error rolling dice: Invalid dice sides" in result


@pytest.mark.asyncio
async def test_dice_roll_negative_sides(mock_random: MagicMock, mock_config: None) -> None:
    with patch("src.tools_and_agents.parse_dice_input", return_value=[(-1, 1)]):
        result: str = await dice_roll("-1d6")
        assert "🎲 Error rolling dice: Invalid dice sides" in result


@pytest.mark.asyncio
async def test_handle_dice_roll(mock_random: MagicMock, mock_config: None) -> None:
    query: str = "roll 2d6"
    result: str = await handle_dice_roll(query)
    assert "🎲 You rolled [5, 5] (total: 10) on 2d6." in result


@pytest.mark.asyncio
async def test_handle_dice_roll_disabled(mock_config: None) -> None:
    with patch("src.tools_and_agents.DICE_ROLLING_ENABLED", False):
        result: str = await handle_dice_roll("roll 2d6")
        assert "Dice rolling is disabled in the configuration." in result


@pytest.mark.asyncio
async def test_web_search(mock_requests: MagicMock, mock_config: None) -> None:
    query: str = "test query"
    mock_response: dict = {"organic_results": [{"snippet": "Test result"}]}
    mock_requests.return_value.json.return_value = mock_response

    result: str = await web_search(query)
    assert "Test result" in result


@pytest.mark.asyncio
async def test_web_search_disabled(mock_requests: MagicMock, mock_config: None) -> None:
    with patch("src.tools_and_agents.WEB_SEARCH_ENABLED", False):
        result: str = await web_search("test query")
        assert "Web search is disabled." in result


@pytest.mark.asyncio
async def test_web_search_api_key_missing(mock_requests: MagicMock, mock_config: None) -> None:
    with patch("src.tools_and_agents.SERPAPI_KEY", None):
        result: str = await web_search("test query")
        assert "SERPAPI_KEY environment variable not set." in result


@pytest.mark.asyncio
async def test_web_search_request_error(mock_requests: MagicMock, mock_config: None) -> None:
    mock_requests.side_effect = requests.exceptions.RequestException("Request failed")
    result: str = await web_search("test query")
    assert "Web search failed: Request failed" in result


@pytest.mark.asyncio
async def test_web_search_no_results(mock_requests: MagicMock, mock_config: None) -> None:
    query: str = "test query"
    mock_response: dict = {"organic_results": []}
    mock_requests.return_value.json.return_value = mock_response

    result: str = await web_search(query)
    assert "No results found." in result


@pytest.mark.asyncio
async def test_web_search_invalid_response(mock_requests: MagicMock, mock_config: None) -> None:
    query: str = "test query"
    mock_response: dict = {"error": "Invalid query"}
    mock_requests.return_value.json.return_value = mock_response

    result: str = await web_search(query)
    assert "Web search failed: Invalid query" in result
