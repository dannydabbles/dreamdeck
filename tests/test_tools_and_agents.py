import pytest
from src.agents.decision_agent import decision_agent
from src.agents.writer_agent import writer_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.dice_agent import dice_roll_agent
from src.agents.web_search_agent import web_search_agent
from langchain_core.messages import HumanMessage, AIMessage
import json


@pytest.mark.asyncio
async def test_decision_agent_roll_action():
    message = HumanMessage(content="roll 2d20")
    response = await decision_agent.ainvoke([message])
    assert isinstance(response, AIMessage)
    assert "dice_roll" in response.content.lower()


@pytest.mark.asyncio
async def test_decision_agent_search_action():
    message = HumanMessage(content="search for information on AI")
    response = await decision_agent.ainvoke([message])
    assert isinstance(response, AIMessage)
    assert "web_search" in response.content.lower()


@pytest.mark.asyncio
async def test_decision_agent_story_action():
    message = HumanMessage(content="continue the story")
    response = await decision_agent.ainvoke([message])
    assert isinstance(response, AIMessage)
    assert "continue_story" in response.content.lower()


@pytest.mark.asyncio
async def test_dice_roll_agent():
    message = HumanMessage(content="roll 2d20")
    response = await dice_roll_agent.ainvoke([message])
    assert isinstance(response, AIMessage)
    assert "🎲 You rolled" in response.content


@pytest.mark.asyncio
async def test_web_search_agent():
    message = HumanMessage(content="search for AI information")
    response = await web_search_agent.ainvoke([message])
    assert isinstance(response, AIMessage)
    assert "Web search" in response.content


@pytest.mark.asyncio
async def test_writer_agent():
    message = HumanMessage(content="continue the story")
    response = await writer_agent.ainvoke([message])
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_storyboard_editor_agent():
    message = HumanMessage(content="generate a storyboard")
    response = await storyboard_editor_agent.ainvoke([message])
    assert isinstance(response, AIMessage)
    assert len(response.content) > 0
