import pytest
from src.config import config

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setenv("APP_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("APP_SERPAPI_KEY", "test-serp-api-key")
