import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState


@pytest.fixture
def adventure_state():
    """Fixture for multi-turn adventure state"""
    return ChatState(
        messages=[
            HumanMessage(content="We enter the dark cave"),
            AIMessage(
                content="You see glowing eyes in the darkness", name="🎭 Storyteller GM"
            ),
            HumanMessage(content="I draw my sword and ready a torch"),
        ],
        thread_id="adventure-1",
        current_persona="Storyteller GM",
    )


@pytest.mark.asyncio
async def test_combat_sequence(adventure_state):
    """Test multi-step combat interaction"""
    # First turn - attack
    adventure_state.messages.append(
        HumanMessage(content="I swing at the creature with my sword!")
    )

    # Mock dice roll and GM response
    mock_responses = [
        AIMessage(content="Roll 1d20+5", name="dice_roll"),
        AIMessage(content="The blade connects!", name="🎭 Storyteller GM"),
    ]

    with patch("src.supervisor.supervisor", AsyncMock(side_effect=[mock_responses])):
        from src.supervisor import supervisor

        response = await supervisor(adventure_state)

        assert len(response) == 2
        assert "Roll 1d20+5" in response[0].content
        assert "blade connects" in response[1].content


@pytest.mark.asyncio
async def test_investigation_sequence(adventure_state):
    """Test investigation with knowledge lookup"""
    adventure_state.messages.append(
        HumanMessage(content="Search the walls for ancient markings")
    )

    mock_responses = [
        AIMessage(content="Found historical records", name="web_search"),
        AIMessage(content="You discover hidden runes...", name="📚 Lorekeeper"),
    ]

    with patch("src.supervisor.supervisor", AsyncMock(side_effect=[mock_responses])):
        from src.supervisor import supervisor

        response = await supervisor(adventure_state)

        assert "historical records" in response[0].content
        assert "hidden runes" in response[1].content
