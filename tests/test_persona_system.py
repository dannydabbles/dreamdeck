import pytest
from unittest.mock import patch, AsyncMock
from src.agents.persona_classifier_agent import persona_classifier_agent, PERSONA_LIST
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage
import chainlit as cl

@pytest.mark.asyncio
async def test_persona_classifier_returns_valid_persona(monkeypatch):
    dummy_state = ChatState(messages=[
        HumanMessage(content="I want to write some code", name="Player")
    ], thread_id="thread1")

    mock_response = AsyncMock()
    mock_response.content = '{"persona": "coder", "reason": "User mentioned code"}'
    with patch("src.agents.persona_classifier_agent.ChatOpenAI.ainvoke", return_value=mock_response):
        result = await persona_classifier_agent(dummy_state)
        assert result["persona"] in PERSONA_LIST
        assert "reason" in result


from src.event_handlers import on_message

@pytest.mark.asyncio
async def test_persona_switch_confirmation(monkeypatch):
    dummy_state = ChatState(messages=[], thread_id="thread1")
    cl.user_session.set("state", dummy_state)
    cl.user_session.set("pending_persona_switch", "therapist")

    msg_yes = cl.Message(content="Yes", author="Player")
    await on_message(msg_yes)
    assert cl.user_session.get("pending_persona_switch") is None
    assert cl.user_session.get("current_persona") == "therapist"

    cl.user_session.set("pending_persona_switch", "coder")
    msg_no = cl.Message(content="No", author="Player")
    await on_message(msg_no)
    assert cl.user_session.get("pending_persona_switch") is None


from src.agents.writer_agent import _generate_story

@pytest.mark.asyncio
async def test_writer_agent_selects_persona_prompt(monkeypatch):
    dummy_state = ChatState(messages=[], thread_id="thread1", current_persona="Secretary")

    async def fake_stream(*args, **kwargs):
        class FakeChunk:
            content = "Test story content"
        yield FakeChunk()
    monkeypatch.setattr("src.agents.writer_agent.ChatOpenAI.astream", fake_stream)

    class DummyMsg:
        content = ""
        id = "msgid"
        async def stream_token(self, chunk): self.content += chunk
        async def send(self): pass
    monkeypatch.setattr("chainlit.Message", lambda **kwargs: DummyMsg())

    result = await _generate_story(dummy_state)
    assert result[0].name.startswith("🗒️ Secretary")


from src.agents.director_agent import _direct_actions

@pytest.mark.asyncio
async def test_director_includes_persona(monkeypatch):
    dummy_state = ChatState(messages=[
        HumanMessage(content="Tell me a story", name="Player")
    ], thread_id="thread1", current_persona="Therapist")

    mock_response = AsyncMock()
    mock_response.content = '{"actions": ["continue_story"]}'
    with patch("src.agents.director_agent.ChatOpenAI.ainvoke", return_value=mock_response):
        actions = await _direct_actions(dummy_state)
        assert isinstance(actions, list)
        assert "continue_story" in actions


from src.workflows import _chat_workflow

@pytest.mark.asyncio
async def test_workflow_filters_avoided_tools(monkeypatch):
    dummy_state = ChatState(messages=[
        HumanMessage(content="I need therapy", name="Player")
    ], thread_id="thread1", current_persona="therapist")

    async def fake_director(state):
        return ["roll", "write"]
    monkeypatch.setattr("src.agents.director_agent.director_agent", fake_director)

    async def fake_dice(state, **kwargs):
        raise RuntimeError("Dice agent should have been skipped")
    monkeypatch.setitem("src.agents.agents_map", "roll", fake_dice)

    async def fake_writer(state, **kwargs):
        return [AIMessage(content="Story", name="writer", metadata={"message_id": "x"})]
    monkeypatch.setitem("src.agents.agents_map", "write", fake_writer)

    cl.user_session.set("vector_memory", None)

    result_state = await _chat_workflow(dummy_state.messages, dummy_state)
    assert any(m.name == "writer" for m in result_state.messages)


@pytest.mark.asyncio
async def test_simulated_conversation_flow(monkeypatch):
    dummy_state = ChatState(messages=[], thread_id="thread1", current_persona="default")

    async def fake_classifier(state, **kwargs):
        cl.user_session.set("suggested_persona", {"persona": "secretary", "reason": "User asked about tasks"})
        return {"persona": "secretary", "reason": "User asked about tasks"}
    monkeypatch.setattr("src.agents.persona_classifier_agent.persona_classifier_agent", fake_classifier)

    async def fake_director(state):
        return ["todo", "write"]
    monkeypatch.setattr("src.agents.director_agent.director_agent", fake_director)

    async def fake_todo(state, **kwargs):
        return [AIMessage(content="Updated TODO", name="todo", metadata={"message_id": "t1"})]
    monkeypatch.setitem("src.agents.agents_map", "todo", fake_todo)

    async def fake_writer(state, **kwargs):
        return [AIMessage(content="Story", name="writer", metadata={"message_id": "w1"})]
    monkeypatch.setitem("src.agents.agents_map", "write", fake_writer)

    cl.user_session.set("vector_memory", None)

    dummy_state.messages.append(HumanMessage(content="Add buy milk", name="Player"))
    result_state = await _chat_workflow(dummy_state.messages, dummy_state)
    names = [m.name for m in result_state.messages]
    assert "todo" in names
    assert "writer" in names
