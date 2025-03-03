import os
import yaml
import logging
from pydantic import BaseModel, ValidationError
from logging.handlers import RotatingFileHandler

# Initialize logging
cl_logger = logging.getLogger("chainlit")

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"
if not os.path.exists(CONFIG_FILE):
    cl_logger.error(f"Configuration file '{CONFIG_FILE}' not found.")
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

with open(CONFIG_FILE, "r") as f:
    config_yaml = yaml.safe_load(f)


class ConfigSchema(BaseModel):
    """Pydantic model for the configuration schema."""
    llm: dict
    prompts: dict
    image_generation_payload: dict
    timeouts: dict
    refusal_list: list
    defaults: dict
    dice: dict
    paths: dict
    openai: dict
    search: dict
    features: dict
    error_handling: dict
    logging: dict
    api: dict
    security: dict
    monitoring: dict
    caching: dict
    agents: dict  # Add agents configuration
    chainlit: dict  # Add chainlit configuration


def load_config() -> ConfigSchema:
    """Load and validate configuration from YAML file.

    Returns:
        ConfigSchema: Validated configuration object
    """
    try:
        config = ConfigSchema(**config_yaml)
    except ValidationError as e:
        cl_logger.error(f"Configuration validation failed: {e}")
        raise

    # Configure logging
    logging.basicConfig(
        level=config.logging.level,
        format=config.logging.format,
        handlers=[
            RotatingFileHandler(
                config.logging.file,
                maxBytes=int(config.logging.max_size),
                backupCount=config.logging.backup_count
            ),
            logging.StreamHandler() if config.logging.console else None
        ],
    )

    # Database configuration with fallbacks
    DATABASE_URL = os.getenv("DATABASE_URL", config.defaults.db_file)
    cl_logger.info(f"Database URL loaded: {DATABASE_URL}")

    # LLM configuration
    llm_config = config.llm
    cl_logger.info(
        f"LLM configuration loaded: "
        f"temperature={llm_config['temperature']}, "
        f"max_tokens={llm_config['max_tokens']}, "
        f"model_name={llm_config['model_name']}, "
        f"streaming={llm_config['streaming']}, "
        f"timeout={llm_config['timeout']}, "
        f"presence_penalty={llm_config['presence_penalty']}, "
        f"frequency_penalty={llm_config['frequency_penalty']}, "
        f"top_p={llm_config['top_p']}, "
        f"verbose={llm_config['verbose']}"
    )

    # Agents configuration
    agents_config = config.agents
    cl_logger.info(
        f"Agents configuration loaded: "
        f"decision_agent (temperature={agents_config.decision_agent.temperature}, "
        f"max_tokens={agents_config.decision_agent.max_tokens}, "
        f"streaming={agents_config.decision_agent.streaming}, "
        f"verbose={agents_config.decision_agent.verbose}), "
        f"writer_agent (temperature={agents_config.writer_agent.temperature}, "
        f"max_tokens={agents_config.writer_agent.max_tokens}, "
        f"streaming={agents_config.writer_agent.streaming}, "
        f"verbose={agents_config.writer_agent.verbose}), "
        f"storyboard_editor_agent (temperature={agents_config.storyboard_editor_agent.temperature}, "
        f"max_tokens={agents_config.storyboard_editor_agent.max_tokens}, "
        f"streaming={agents_config.storyboard_editor_agent.streaming}, "
        f"verbose={agents_config.storyboard_editor_agent.verbose})"
    )

    # Image generation payload
    image_generation_payload = config.image_generation_payload
    cl_logger.info(
        f"Image generation payload loaded: "
        f"negative_prompt={image_generation_payload['negative_prompt']}, "
        f"steps={image_generation_payload['steps']}, "
        f"sampler_name={image_generation_payload['sampler_name']}, "
        f"scheduler={image_generation_payload['scheduler']}, "
        f"cfg_scale={image_generation_payload['cfg_scale']}, "
        f"width={image_generation_payload['width']}, "
        f"height={image_generation_payload['height']}, "
        f"hr_upscaler={image_generation_payload['hr_upscaler']}, "
        f"denoising_strength={image_generation_payload['denoising_strength']}, "
        f"hr_second_pass_steps={image_generation_payload['hr_second_pass_steps']}"
    )

    # Timeouts
    IMAGE_GENERATION_TIMEOUT = config.timeouts.image_generation_timeout
    cl_logger.info(f"Image generation timeout loaded: {IMAGE_GENERATION_TIMEOUT}")

    # Token limits
    LLM_MAX_TOKENS = llm_config.max_tokens
    cl_logger.info(f"LLM max tokens loaded: {LLM_MAX_TOKENS}")

    # Refusal list
    REFUSAL_LIST = config.refusal_list
    cl_logger.info(f"Refusal list loaded: {REFUSAL_LIST}")

    # Defaults
    DB_FILE = config.defaults.db_file
    cl_logger.info(f"Default DB file loaded: {DB_FILE}")

    # Dice settings
    DICE_SIDES = config.dice.sides  # Default to d20
    cl_logger.info(f"Default dice sides loaded: {DICE_SIDES}")

    # Knowledge directory
    KNOWLEDGE_DIRECTORY = config.paths.knowledge
    cl_logger.info(f"Knowledge directory loaded: {KNOWLEDGE_DIRECTORY}")

    # LLM settings
    LLM_SETTINGS = config.llm
    LLM_CHUNK_SIZE = LLM_SETTINGS.get('chunk_size', 1000)  # Default to 1000 if not set
    LLM_TEMPERATURE = LLM_SETTINGS.temperature
    LLM_MODEL_NAME = LLM_SETTINGS.model_name
    cl_logger.info(
        f"LLM settings loaded: chunk_size={LLM_CHUNK_SIZE}, "
        f"temperature={LLM_TEMPERATURE}, "
        f"model_name={LLM_MODEL_NAME}"
    )

    # Image settings
    IMAGE_SETTINGS = config.image_settings
    NUM_IMAGE_PROMPTS = IMAGE_SETTINGS.num_image_prompts
    cl_logger.info(f"Image settings loaded: num_image_prompts={NUM_IMAGE_PROMPTS}")

    # OpenAI settings
    OPENAI_BASE_URL = config.openai.base_url
    OPENAI_API_KEY = config.openai.api_key
    cl_logger.info(
        f"OpenAI settings loaded: base_url={OPENAI_BASE_URL}, api_key={OPENAI_API_KEY}"
    )

    # Search settings
    SERPAPI_KEY = config.search.serpapi_key
    cl_logger.info(f"Search settings loaded: serpapi_key={SERPAPI_KEY}")

    # Chainlit Starters
    CHAINLIT_STARTERS = config.chainlit.get('starters', [])
    cl_logger.info(f"Chainlit starters loaded: {CHAINLIT_STARTERS}")

    return config
