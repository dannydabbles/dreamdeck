import pytest
from unittest.mock import patch, AsyncMock
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage

@pytest.mark.asyncio
async def test_character_agent():
    from src.agents.character_agent import _character
    with patch("src.agents.character_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke, \
         patch("src.agents.character_agent.cl.user_session.get", return_value={}):
        mock_ainvoke.return_value = AIMessage(content="Character info", name="character")
        state = ChatState(messages=[HumanMessage(content="Describe hero")], thread_id="t")
        result = await _character(state)
        assert result[0].name == "character"
        assert "Character" in result[0].content or result[0].content

@pytest.mark.asyncio
async def test_lore_agent():
    from src.agents.lore_agent import _lore
    with patch("src.agents.lore_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke, \
         patch("src.agents.lore_agent.cl.user_session.get", return_value={}):
        mock_ainvoke.return_value = AIMessage(content="Lore info", name="lore")
        state = ChatState(messages=[HumanMessage(content="Tell me lore")], thread_id="t")
        result = await _lore(state)
        assert result[0].name == "lore"
        assert "Lore" in result[0].content or result[0].content

@pytest.mark.asyncio
async def test_puzzle_agent():
    from src.agents.puzzle_agent import _puzzle
    with patch("src.agents.puzzle_agent.ChatOpenAI.ainvoke", new_callable=AsyncMock) as mock_ainvoke, \
         patch("src.agents.puzzle_agent.cl.user_session.get", return_value={}):
        mock_ainvoke.return_value = AIMessage(content="Puzzle info", name="puzzle")
        state = ChatState(messages=[HumanMessage(content="Give me a puzzle")], thread_id="t")
        result = await _puzzle(state)
        assert result[0].name == "puzzle"
        assert "Puzzle" in result[0].content or result[0].content
