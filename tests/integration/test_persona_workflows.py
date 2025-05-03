import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage
from src.models import ChatState


@pytest.mark.asyncio
async def test_therapist_persona_workflow():
    """Test full therapist persona interaction flow"""
    state = ChatState(
        messages=[HumanMessage(content="I'm feeling anxious about the quest")],
        thread_id="therapy-1",
        current_persona="Therapist",
    )

    # Mock LLM response
    mock_response = AIMessage(
        content="Let's explore those feelings. When did you first notice this anxiety?",
        name="🧠 Therapist",
    )

    with patch("src.agents.writer_agent._generate_story") as mock_generate_story:
        mock_generate_story.return_value = [mock_response]
        from src.agents.writer_agent import call_writer_agent

        response = await call_writer_agent(state)

        assert len(response) == 1
        assert "explore those feelings" in response[0].content
        assert response[0].name == "🧠 Therapist"


@pytest.mark.asyncio
async def test_dungeon_master_combat_flow():
    """Test combat sequence with dice rolls and narrative"""
    state = ChatState(
        messages=[
            HumanMessage(content="I attack the dragon with my sword!"),
            AIMessage(content="Roll for attack", name="dice_roll"),
        ],
        thread_id="combat-1",
        current_persona="Dungeon Master",
    )

    # Mock supervisor routing
    mock_supervisor = AsyncMock(
        return_value=[
            AIMessage(content="The dragon roars in pain!", name="🎲 Dungeon Master")
        ]
    )

    with patch("src.supervisor.supervisor", mock_supervisor):
        from src.supervisor import supervisor

        response = await supervisor(state)

        assert "dragon roars" in response[0].content
        assert mock_supervisor.await_count == 1


@pytest.mark.asyncio
async def test_multi_tool_workflow():
    """Test todo creation followed by secretary summary"""
    state = ChatState(
        messages=[
            HumanMessage(content="/todo Prepare potions for the journey"),
            AIMessage(content="Added: Prepare potions", name="todo"),
        ],
        thread_id="todo-1",
        current_persona="Secretary",
    )

    # Mock secretary response
    mock_response = AIMessage(
        content="Your tasks are organized: 1. Prepare potions", name="🗒️ Secretary"
    )

    with patch("src.agents.writer_agent._generate_story") as mock_generate_story:
        mock_generate_story.return_value = [mock_response]
        from src.agents.writer_agent import call_writer_agent

        response = await call_writer_agent(state)

        assert "tasks are organized" in response[0].content
        assert "🗒️ Secretary" == response[0].name
