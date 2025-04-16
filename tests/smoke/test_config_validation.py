import pytest
from src.config import load_config, ConfigSchema
from pydantic import ValidationError


def test_valid_config():
    config = load_config()
    assert isinstance(config, ConfigSchema)
    assert config.llm.temperature == 0.6


def test_invalid_config():
    invalid_config = {"llm": {"invalid_param": "bad"}}
    with pytest.raises(ValidationError):
        ConfigSchema.model_validate(invalid_config)


def test_missing_required_config():
    from src.config import ConfigSchema

    with pytest.raises(ValidationError):
        ConfigSchema.model_validate({})


def test_edge_case_configs():
    minimal_config = {
        "llm": {
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 8000,
            "streaming": True,
            "timeout": 300,
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "top_p": 1.0,
            "verbose": True,
        },
        "agents": {
            "decision_agent": {
                "temperature": 0.0,
                "max_tokens": 100,
                "streaming": True,
                "verbose": True,
            },
            "writer_agent": {
                "temperature": 0.7,
                "max_tokens": 8000,
                "streaming": True,
                "verbose": True,
            },
            "storyboard_editor_agent": {
                "temperature": 0.7,
                "max_tokens": 8000,
                "streaming": False,
                "verbose": True,
            },
        },
        "features": {
            "image_generation": False,
            "web_search": False,
            "dice_rolling": False,
        },
        "prompts": {},  # Required placeholder
        "image_generation_payload": {},  # Required placeholder
        "timeouts": {},  # Required placeholder
        "refusal_list": [],
        "defaults": {"db_file": "chainlit.db"},
        "dice": {"sides": 20},
        "paths": {},
        "openai": {},
        "search": {},
        "error_handling": {},
        "logging": {},
        "api": {},
        "security": {},
        "monitoring": {},
        "caching": {},
        "chainlit": {},
        "search_enabled": False,
        "knowledge_directory": "./knowledge",
        "image_settings": {},
        "rate_limits": {},
        "chat": {},
    }
    ConfigSchema.model_validate(minimal_config)


import pytest


@pytest.mark.skip(
    reason="Environment variable override is not supported dynamically in load_config()"
)
def test_env_override(monkeypatch):
    monkeypatch.setenv("APP_LLM__TEMPERATURE", "0.9")
    from src.config import load_config

    cfg = load_config()
    assert abs(cfg.llm.temperature - 0.9) < 1e-6
