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
    # Test minimal valid config
    minimal_config = {
        "llm": {"temperature": 0.0},
        "agents": {
            "decision_agent": {"temperature": 0.0}
        },
        "features": {"image_generation": False}
    }
    ConfigSchema.model_validate(minimal_config)
