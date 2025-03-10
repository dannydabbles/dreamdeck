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

def test_required_fields():
    minimal_config = {
        "llm": {"model_name": "gpt-3.5-turbo"},
        "features": {"image_generation": True},
    }
    ConfigSchema.model_validate(minimal_config)
