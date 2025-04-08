import pytest
from src.models import ChatState
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

def test_chat_state_initialization():
    state = ChatState(thread_id="test-thread")
    assert state.messages == []
    assert state.thread_id == "test-thread"
    assert state.error_count == 0
    assert state.memories == []

def test_chat_state_get_memories_str():
    state = ChatState(thread_id="test", memories=["memory1", "memory2", "memory3", "memory4"])
    assert state.get_memories_str() == "memory1\nmemory2\nmemory3"
    state_empty = ChatState(thread_id="test", memories=[])
    assert state_empty.get_memories_str() == ""
    state_less_than_3 = ChatState(thread_id="test", memories=["m1", "m2"])
    assert state_less_than_3.get_memories_str() == "m1\nm2"

def test_chat_state_get_last_human_message():
    state = ChatState(thread_id="test", messages=[
        HumanMessage(content="Hello", name="Player"),
        AIMessage(content="Hi there", name="Game Master"),
        HumanMessage(content="How are you?", name="Player")
    ])
    last_human = state.get_last_human_message()
    assert isinstance(last_human, HumanMessage)
    assert last_human.content == "How are you?"

    state_no_human = ChatState(thread_id="test", messages=[
        AIMessage(content="Hi there", name="Game Master")
    ])
    assert state_no_human.get_last_human_message() is None

    state_empty = ChatState(thread_id="test", messages=[])
    assert state_empty.get_last_human_message() is None

def test_chat_state_get_recent_history_str():
    messages = [
        SystemMessage(content="System init"),
        HumanMessage(content="Msg 1", name="Player"),
        AIMessage(content="Msg 2", name="Game Master"),
        ToolMessage(content="Tool output", tool_call_id="123"),
        HumanMessage(content="Msg 3", name="Player"),
        AIMessage(content="Msg 4", name="Game Master")
    ]
    state = ChatState(thread_id="test", messages=messages)

    # Test default (last 500, filters non-chat)
    history_str = state.get_recent_history_str()
    expected_str = "Player: Msg 1\nGame Master: Msg 2\nPlayer: Msg 3\nGame Master: Msg 4"
    assert history_str == expected_str

    # Test with smaller n
    history_str_n2 = state.get_recent_history_str(n=2)
    # Note: n=2 applies *before* filtering, so it takes last 2 messages overall,
    # then filters them. In this case, HumanMsg3 and AIMsg4 remain.
    expected_str_n2 = "Player: Msg 3\nGame Master: Msg 4"
    assert history_str_n2 == expected_str_n2

    state_empty = ChatState(thread_id="test", messages=[])
    assert state_empty.get_recent_history_str() == ""

def test_chat_state_get_tool_results_str():
    messages = [
        HumanMessage(content="Do something", name="Player"),
        AIMessage(content="Rolling dice...", name="dice_roll"), # Tool result 1
        AIMessage(content="The result is 5", name="Game Master"),
        HumanMessage(content="Search for cats", name="Player"),
        AIMessage(content="Searching...", name="web_search"), # Tool result 2 (most recent)
        AIMessage(content="Found cats", name="Game Master"),
    ]
    state = ChatState(thread_id="test", messages=messages)
    tool_str = state.get_tool_results_str()
    # Should get the most recent tool result (web_search)
    assert tool_str == "web_search: Searching..."

    messages_only_dice = [
        HumanMessage(content="Roll", name="Player"),
        AIMessage(content="Rolling...", name="dice_roll"),
        AIMessage(content="Result 7", name="Game Master"),
    ]
    state_dice = ChatState(thread_id="test", messages=messages_only_dice)
    assert state_dice.get_tool_results_str() == "dice_roll: Rolling..."

    messages_no_tools = [
        HumanMessage(content="Hi", name="Player"),
        AIMessage(content="Hello", name="Game Master"),
    ]
    state_no_tools = ChatState(thread_id="test", messages=messages_no_tools)
    assert state_no_tools.get_tool_results_str() == ""

    state_empty = ChatState(thread_id="test", messages=[])
    assert state_empty.get_tool_results_str() == ""

def test_chat_state_get_recent_history():
    messages = [HumanMessage(content=f"Msg {i}") for i in range(10)]
    state = ChatState(thread_id="test", messages=messages)

    # Default n=500 (gets all 10)
    recent = state.get_recent_history()
    assert len(recent) == 10
    assert recent == messages

    # n=3
    recent_n3 = state.get_recent_history(n=3)
    assert len(recent_n3) == 3
    assert recent_n3 == messages[-3:]

    # n > len(messages)
    recent_n15 = state.get_recent_history(n=15)
    assert len(recent_n15) == 10
    assert recent_n15 == messages

    state_empty = ChatState(thread_id="test", messages=[])
    assert state_empty.get_recent_history() == []
    assert state_empty.get_recent_history(n=5) == []
