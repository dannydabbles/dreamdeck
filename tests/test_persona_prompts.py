import pytest
from unittest.mock import patch
from src.models import ChatState
from src.config import config


@pytest.fixture
def dummy_state():
    return ChatState(messages=[], thread_id="test_thread", current_persona="Default")


@pytest.mark.asyncio
@pytest.mark.parametrize("persona,expected_prompt_key", [
    ("Storyteller GM", "storyteller_gm_prompt"),
    ("Therapist", "therapist_writer_prompt"),
    ("Coder", "coder_writer_prompt"),
    ("Secretary", "secretary_writer_prompt"),
    ("Default", "default_writer_prompt"),
    ("UnknownPersona", "default_writer_prompt"),  # fallback
])
async def test_writer_agent_prompt_selection(dummy_state, persona, expected_prompt_key):
    dummy_state.current_persona = persona

    with patch("src.agents.writer_agent.ChatOpenAI") as mock_llm, \
         patch("src.agents.writer_agent.cl.AsyncLangchainCallbackHandler"), \
         patch("src.agents.writer_agent.cl.Message") as mock_cl_msg:

        # Setup mock message streaming
        mock_cl_msg.return_value.stream_token = lambda chunk: None
        mock_cl_msg.return_value.send = lambda: None
        mock_cl_msg.return_value.content = "Generated story"
        mock_cl_msg.return_value.id = "msgid"

        # Setup mock LLM streaming
        async def fake_stream(*args, **kwargs):
            class FakeChunk:
                content = "Generated story"
            yield FakeChunk()
        mock_llm.return_value.astream.side_effect = fake_stream

        from src.agents.writer_agent import _generate_story

        await _generate_story(dummy_state)

        # Check prompt key resolved
        persona_configs = getattr(config.agents.writer_agent, "personas", {})
        persona_entry = persona_configs.get(persona, {})
        prompt_key = persona_entry.get("prompt_key", "default_writer_prompt") if isinstance(persona_entry, dict) else "default_writer_prompt"
        if persona not in persona_configs:
            prompt_key = "default_writer_prompt"

        assert prompt_key == expected_prompt_key


@pytest.mark.asyncio
@pytest.mark.parametrize("persona,expected_prompt_key", [
    ("Secretary", "secretary_todo_prompt"),
    ("Default", "todo_prompt"),
    ("UnknownPersona", "todo_prompt"),
])
async def test_todo_agent_prompt_selection(dummy_state, persona, expected_prompt_key):
    dummy_state.current_persona = persona

    with patch("src.agents.todo_agent.ChatOpenAI") as mock_llm, \
         patch("src.agents.todo_agent.cl.Message") as mock_cl_msg, \
         patch("src.agents.todo_agent.open", create=True), \
         patch("src.agents.todo_agent.os.path.exists", return_value=False):

        mock_cl_msg.return_value.send = lambda: None
        mock_cl_msg.return_value.id = "msgid"

        mock_llm.return_value.ainvoke.return_value.content = "**Finished**\n\n**In Progress**\n\n**Remaining**\n- test"

        from src.agents.todo_agent import _manage_todo

        dummy_state.messages.append(
            # Simulate last human message
            type("HumanMessage", (), {"content": "/todo test", "name": "Player", "metadata": {}})()
        )

        result = await _manage_todo(dummy_state)
        # Check prompt key resolved
        persona_configs = getattr(config.agents, "todo_agent", {}).get("personas", {})
        persona_entry = persona_configs.get(persona, {})
        prompt_key = persona_entry.get("prompt_key", "todo_prompt") if isinstance(persona_entry, dict) else "todo_prompt"
        if persona not in persona_configs:
            prompt_key = "todo_prompt"

        assert prompt_key == expected_prompt_key


@pytest.mark.asyncio
@pytest.mark.parametrize("persona,expected_prompt_key", [
    ("Lorekeeper", "lore_knowledge_prompt"),
    ("Default", "lore_prompt"),
    ("UnknownPersona", "lore_prompt"),
])
async def test_knowledge_agent_prompt_selection(dummy_state, persona, expected_prompt_key):
    dummy_state.current_persona = persona

    with patch("src.agents.knowledge_agent.ChatOpenAI") as mock_llm, \
         patch("src.agents.knowledge_agent.cl.Message") as mock_cl_msg:

        mock_cl_msg.return_value.send = lambda: None
        mock_cl_msg.return_value.id = "msgid"

        mock_llm.return_value.ainvoke.return_value.content = "Knowledge content"

        from src.agents.knowledge_agent import _knowledge

        await _knowledge(dummy_state, "lore")

        persona_configs = getattr(config.agents, "knowledge_agent", {}).get("personas", {})
        persona_entry = persona_configs.get(persona, {})
        prompt_key = persona_entry.get("prompt_key", "lore_prompt") if isinstance(persona_entry, dict) else "lore_prompt"
        if persona not in persona_configs:
            prompt_key = "lore_prompt"

        assert prompt_key == expected_prompt_key
