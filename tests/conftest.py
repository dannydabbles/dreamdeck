import pytest
from src.config import config, DefaultsConfig, LlmConfig, DiceConfig, FeatureConfig, DecisionAgentConfig, WriterAgentConfig, StoryboardEditorAgentConfig, AgentsConfig

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    monkeypatch.setattr("src.config.config", 
        config.ConfigSchema(
            defaults=DefaultsConfig(db_file=":memory:?check_same_thread=False"),
            llm=LlmConfig(temperature=0.6, max_tokens=8000, model_name="gpt-3.5-turbo", streaming=True, timeout=180, presence_penalty=0.1, frequency_penalty=0.1, top_p=1.0, verbose=True),
            prompts={},
            image_generation_payload={},
            timeouts={},
            refusal_list=[],
            dice=DiceConfig(sides=20),
            paths={},
            openai={},
            search={},
            features=FeatureConfig(dice_rolling=True, web_search=False, image_generation=True),
            error_handling={},
            logging={},
            api={},
            security={},
            monitoring={},
            caching={},
            agents=AgentsConfig(
                decision_agent=DecisionAgentConfig(temperature=0.2, max_tokens=100, streaming=True, verbose=True),
                writer_agent=WriterAgentConfig(temperature=0.7, max_tokens=8000, streaming=True, verbose=True),
                storyboard_editor_agent=StoryboardEditorAgentConfig(temperature=0.7, max_tokens=8000, streaming=False, verbose=True)
            ),
            chainlit={},
            search_enabled=False,
            knowledge_directory="",
            image_settings={},
            storyboard_generation_prompt_prefix="",
            storyboard_generation_prompt_postfix=""
        )
    )
    monkeypatch.setenv("APP_DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("APP_SERPAPI_KEY", "test-serp-api-key")
