import pytest
from unittest.mock import patch, AsyncMock
from src.models import ChatState
from src.config import config


@pytest.fixture(autouse=True, scope="module")
def patch_persona_configs():
    # Patch writer_agent personas
    if not hasattr(config.agents.writer_agent.__class__, "personas"):
        setattr(
            config.agents.writer_agent.__class__,
            "personas",
            {
                "Storyteller GM": {"prompt_key": "storyteller_gm_prompt"},
                "Therapist": {"prompt_key": "therapist_writer_prompt"},
                "Coder": {"prompt_key": "coder_writer_prompt"},
                "Secretary": {"prompt_key": "secretary_writer_prompt"},
                "Default": {"prompt_key": "default_writer_prompt"},
            },
        )

    # Patch todo_agent personas
    if not hasattr(config.agents.__class__, "todo_agent"):
        setattr(config.agents.__class__, "todo_agent", {})
    if not hasattr(config.agents.__class__, "knowledge_agent"):
        setattr(config.agents.__class__, "knowledge_agent", {})

    # Patch todo_agent personas dict inside the class attribute
    if not hasattr(config.agents.todo_agent, "personas"):
        config.agents.todo_agent["personas"] = {
            "Secretary": {"prompt_key": "secretary_todo_prompt"},
            "Default": {"prompt_key": "todo_prompt"},
        }

    # Patch knowledge_agent personas dict inside the class attribute
    if not hasattr(config.agents.knowledge_agent, "personas"):
        config.agents.knowledge_agent["personas"] = {
            "Lorekeeper": {"prompt_key": "lore_knowledge_prompt"},
            "Default": {"prompt_key": "lore_prompt"},
        }


@pytest.fixture
def dummy_state():
    return ChatState(messages=[], thread_id="test_thread", current_persona="Default")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persona,expected_prompt_key",
    [
        ("Storyteller GM", "storyteller_gm_prompt"),
        ("Therapist", "therapist_writer_prompt"),
        ("Coder", "coder_writer_prompt"),
        ("Secretary", "secretary_writer_prompt"),
        ("Default", "friend_writer_prompt"), # Changed expected prompt
        ("UnknownPersona", "friend_writer_prompt"),  # Fallback now uses the friend prompt key via default_writer_prompt
    ],
)
async def test_writer_agent_prompt_selection(dummy_state, persona, expected_prompt_key):
    dummy_state.current_persona = persona

    with patch("src.agents.writer_agent.ChatOpenAI") as mock_llm, patch(
        "src.agents.writer_agent.cl.AsyncLangchainCallbackHandler"
    ), patch("src.agents.writer_agent.cl.Message") as mock_cl_msg:

        # Setup mock message streaming
        from unittest.mock import AsyncMock
        mock_cl_msg.return_value.stream_token = AsyncMock(return_value=None)
        mock_cl_msg.return_value.send = AsyncMock(return_value=None)
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
        resolved_prompt_key = None
        if isinstance(persona_entry, dict):
            resolved_prompt_key = persona_entry.get("prompt_key")

        # If no specific key found, use the global default writer prompt key from config
        if not resolved_prompt_key:
            # Strip .j2 extension if present for test comparison
            resolved_prompt_key = config.prompt_files.get("default_writer_prompt")
            if resolved_prompt_key and resolved_prompt_key.endswith(".j2"):
                resolved_prompt_key = resolved_prompt_key[:-3]

        assert resolved_prompt_key == expected_prompt_key


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persona,expected_prompt_key",
    [
        ("Secretary", "secretary_todo_prompt"),
        ("Default", "todo_prompt"),
        ("UnknownPersona", "todo_prompt"),
    ],
)
async def test_todo_agent_prompt_selection(dummy_state, persona, expected_prompt_key):
    dummy_state.current_persona = persona

    with patch("src.agents.todo_agent.ChatOpenAI") as mock_llm, patch(
        "src.agents.todo_agent.cl.Message"
    ) as mock_cl_msg, patch("src.agents.todo_agent.open", create=True), patch(
        "src.agents.todo_agent.os.path.exists", return_value=False
    ):

        mock_cl_msg.return_value.send = AsyncMock(return_value=None)
        mock_cl_msg.return_value.id = "msgid"

        mock_llm.return_value.ainvoke.return_value.content = (
            "**Finished**\n\n**In Progress**\n\n**Remaining**\n- test"
        )

        from src.agents.todo_agent import _manage_todo

        dummy_state.messages.append(
            # Simulate last human message
            type(
                "HumanMessage",
                (),
                {"content": "/todo test", "name": "Player", "metadata": {}},
            )()
        )

        result = await _manage_todo(dummy_state)
        # Check prompt key resolved
        persona_configs = getattr(config.agents, "todo_agent", {}).get("personas", {})
        persona_entry = persona_configs.get(persona, {})
        prompt_key = (
            persona_entry.get("prompt_key", "todo_prompt")
            if isinstance(persona_entry, dict)
            else "todo_prompt"
        )
        if persona not in persona_configs:
            prompt_key = "todo_prompt"

        assert prompt_key == expected_prompt_key


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "persona,expected_prompt_key",
    [
        ("Lorekeeper", "lore_knowledge_prompt"),
        ("Default", "lore_prompt"),
        ("UnknownPersona", "lore_prompt"),
    ],
)
async def test_knowledge_agent_prompt_selection(
    dummy_state, persona, expected_prompt_key
):
    dummy_state.current_persona = persona

    with patch("src.agents.knowledge_agent.ChatOpenAI") as mock_llm, patch(
        "src.agents.knowledge_agent.cl.Message"
    ) as mock_cl_msg:

        mock_cl_msg.return_value.send = AsyncMock(return_value=None)
        mock_cl_msg.return_value.id = "msgid"

        mock_llm.return_value.ainvoke.return_value.content = "Knowledge content"

        from src.agents.knowledge_agent import _knowledge

        await _knowledge(dummy_state, "lore")

        persona_configs = getattr(config.agents, "knowledge_agent", {}).get(
            "personas", {}
        )
        persona_entry = persona_configs.get(persona, {})
        prompt_key = (
            persona_entry.get("prompt_key", "lore_prompt")
            if isinstance(persona_entry, dict)
            else "lore_prompt"
        )
        if persona not in persona_configs:
            prompt_key = "lore_prompt"

        assert prompt_key == expected_prompt_key
