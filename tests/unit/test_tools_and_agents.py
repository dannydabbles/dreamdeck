import pytest
from src.tools_and_agents import dice_roll, parse_dice_input, web_search
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_random():
    with patch('random.randint') as mock_rand:
        mock_rand.return_value = 5
        yield mock_rand

def test_parse_dice_input_single_die():
    input_str = "d20"
    result = parse_dice_input(input_str)
    assert result == [(20, 1)]

def test_parse_dice_input_multiple_dice():
    input_str = "2d6"
    result = parse_dice_input(input_str)
    assert result == [(6, 2)]

def test_parse_dice_input_complex():
    input_str = "3d4 + 1d8"
    result = parse_dice_input(input_str)
    assert result == [(4, 3), (8, 1)]

def test_parse_dice_input_invalid():
    input_str = "invalid"
    with pytest.raises(ValueError, match="Invalid dice specification"):
        parse_dice_input(input_str)

@pytest.mark.asyncio
async def test_dice_roll_single_die(mock_random):
    result = await dice_roll()
    assert "🎲 You rolled a 5 on a 20-sided die." in result

@pytest.mark.asyncio
async def test_dice_roll_multiple_dice(mock_random):
    result = await dice_roll("2d6")
    assert "🎲 You rolled [5, 5] (total: 10) on 2d6." in result

@pytest.mark.asyncio
async def test_dice_roll_invalid_input():
    result = await dice_roll("invalid")
    assert "🎲 Error rolling dice: Invalid dice specification" in result

@pytest.mark.asyncio
async def test_dice_roll_disabled():
    with patch("src.tools_and_agents.DICE_ROLLING_ENABLED", False):
        result = await dice_roll()
        assert "Dice rolling is disabled in the configuration." in result

@pytest.mark.asyncio
async def test_web_search(mock_requests):
    query = "test query"
    mock_response = {"organic_results": [{"snippet": "Test result"}]}
    mock_requests.get.return_value.json.return_value = mock_response

    result = await web_search(query)
    assert "Test result" in result

@pytest.mark.asyncio
async def test_web_search_disabled(mock_requests):
    with patch("src.tools_and_agents.WEB_SEARCH_ENABLED", False):
        result = await web_search("test query")
        assert "Web search is disabled." in result

@pytest.mark.asyncio
async def test_web_search_api_key_missing(mock_requests):
    with patch("src.tools_and_agents.SERPAPI_KEY", None):
        result = await web_search("test query")
        assert "SERPAPI_KEY environment variable not set." in result

@pytest.mark.asyncio
async def test_web_search_request_error(mock_requests):
    mock_requests.get.side_effect = MagicMock(side_effect=requests.exceptions.RequestException("Request failed"))
    result = await web_search("test query")
    assert "Web search failed: Request failed" in result
