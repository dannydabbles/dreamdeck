import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.models import ChatState


def test_chat_state_initialization():
    state = ChatState(thread_id="test-thread")
    assert state.messages == []
    assert state.thread_id == "test-thread"
    assert state.error_count == 0
    assert state.memories == []


def test_chat_state_get_memories_str():
    state = ChatState(thread_id="test", memories=["m1", "m2", "m3", "m4", "m5"])
    result = state.get_memories_str()
    assert result == "m1\nm2\nm3"

    state_empty = ChatState(thread_id="test", memories=[])
    assert state_empty.get_memories_str() == ""

    state_few = ChatState(thread_id="test", memories=["x", "y"])
    assert state_few.get_memories_str() == "x\ny"


def test_chat_state_get_recent_history_str():
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi", name="Player"),
        AIMessage(content="hello", name="Game Master"),
        ToolMessage(content="tool", tool_call_id="t1"),
        HumanMessage(content="bye", name="Player"),
        AIMessage(content="see ya", name="Game Master"),
    ]
    state = ChatState(thread_id="t", messages=msgs)
    s = state.get_recent_history_str()
    assert "Player: hi" in s
    assert "Game Master: hello" in s
    assert "Player: bye" in s
    assert "Game Master: see ya" in s
    assert "sys" not in s
    assert "tool" not in s


def test_chat_state_get_tool_results_str():
    msgs = [
        HumanMessage(content="hi", name="Player"),
        AIMessage(content="rolling", name="dice_roll"),
        AIMessage(content="story", name="Game Master"),
        AIMessage(content="searching", name="web_search"),
    ]
    state = ChatState(thread_id="t", messages=msgs)
    s = state.get_tool_results_str()
    assert s.startswith("web_search: searching")


def test_chat_state_get_recent_history():
    msgs = [HumanMessage(content=f"m{i}") for i in range(5)]
    state = ChatState(thread_id="t", messages=msgs)
    recent = state.get_recent_history(n=3)
    assert recent == msgs[-3:]


def test_increment_error_count():
    state = ChatState(thread_id="t")
    assert state.error_count == 0
    state.increment_error_count()
    assert state.error_count == 1
