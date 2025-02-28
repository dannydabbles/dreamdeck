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
    chainlit: dict
    logging: dict
    error_handling: dict
    api: dict
    features: dict
    rate_limits: dict
    security: dict
    monitoring: dict
    caching: dict


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

    # Database configuration with fallbacks
    DATABASE_URL = os.getenv("DATABASE_URL", config.defaults.db_file)
    STABLE_DIFFUSION_API_URL = os.getenv(
        "STABLE_DIFFUSION_API_URL", "http://localhost:7860"
    )
    cl_logger.info(f"Database URL loaded: {DATABASE_URL}")

    # LLM configuration
    LLM_TEMPERATURE = config.llm.temperature
    LLM_MAX_TOKENS = config.llm.max_tokens
    LLM_MODEL_NAME = config.llm.model_name
    LLM_STREAMING = config.llm.streaming
    LLM_TIMEOUT = config.llm.timeout
    LLM_PRESENCE_PENALTY = config.llm.presence_penalty
    LLM_FREQUENCY_PENALTY = config.llm.frequency_penalty
    LLM_TOP_P = config.llm.top_p
    LLM_VERBOSE = config.llm.verbose
    cl_logger.info(
        f"LLM configuration loaded: temperature={LLM_TEMPERATURE}, max_tokens={LLM_MAX_TOKENS}, model_name={LLM_MODEL_NAME}, streaming={LLM_STREAMING}, timeout={LLM_TIMEOUT}, presence_penalty={LLM_PRESENCE_PENALTY}, frequency_penalty={LLM_FREQUENCY_PENALTY}, top_p={LLM_TOP_P}, verbose={LLM_VERBOSE}"
    )

    # Agents configuration
    DECISION_AGENT_TEMPERATURE = config.agents.decision_agent.temperature
    DECISION_AGENT_MAX_TOKENS = config.agents.decision_agent.max_tokens
    DECISION_AGENT_STREAMING = config.agents.decision_agent.streaming
    DECISION_AGENT_VERBOSE = config.agents.decision_agent.verbose
    WRITER_AGENT_TEMPERATURE = config.agents.writer_agent.temperature
    WRITER_AGENT_MAX_TOKENS = config.agents.writer_agent.max_tokens
    WRITER_AGENT_STREAMING = config.agents.writer_agent.streaming
    WRITER_AGENT_VERBOSE = config.agents.writer_agent.verbose
    STORYBOARD_EDITOR_AGENT_TEMPERATURE = (
        config.agents.storyboard_editor_agent.temperature
    )
    STORYBOARD_EDITOR_AGENT_MAX_TOKENS = (
        config.agents.storyboard_editor_agent.max_tokens
    )
    STORYBOARD_EDITOR_AGENT_STREAMING = config.agents.storyboard_editor_agent.streaming
    STORYBOARD_EDITOR_AGENT_VERBOSE = config.agents.storyboard_editor_agent.verbose
    cl_logger.info(
        f"Agents configuration loaded: decision_agent (temperature={DECISION_AGENT_TEMPERATURE}, max_tokens={DECISION_AGENT_MAX_TOKENS}, streaming={DECISION_AGENT_STREAMING}, verbose={DECISION_AGENT_VERBOSE}), writer_agent (temperature={WRITER_AGENT_TEMPERATURE}, max_tokens={WRITER_AGENT_MAX_TOKENS}, streaming={WRITER_AGENT_STREAMING}, verbose={WRITER_AGENT_VERBOSE}), storyboard_editor_agent (temperature={STORYBOARD_EDITOR_AGENT_TEMPERATURE}, max_tokens={STORYBOARD_EDITOR_AGENT_MAX_TOKENS}, streaming={STORYBOARD_EDITOR_AGENT_STREAMING}, verbose={STORYBOARD_EDITOR_AGENT_VERBOSE})"
    )

    # Prompts
    AI_WRITER_PROMPT = config.prompts.ai_writer_prompt
    STORYBOARD_GENERATION_PROMPT = config.prompts.storyboard_generation_prompt
    STORYBOARD_GENERATION_PROMPT_PREFIX = (
        config.prompts.storyboard_generation_prompt_prefix
    )
    STORYBOARD_GENERATION_PROMPT_POSTFIX = (
        config.prompts.storyboard_generation_prompt_postfix
    )
    DECISION_PROMPT = config.prompts.decision_prompt  # Load the new prompt

    # Image generation payload
    IMAGE_GENERATION_PAYLOAD = config.image_generation_payload
    NEGATIVE_PROMPT = IMAGE_GENERATION_PAYLOAD.negative_prompt
    STEPS = IMAGE_GENERATION_PAYLOAD.steps
    SAMPLER_NAME = IMAGE_GENERATION_PAYLOAD.sampler_name
    SCHEDULER = IMAGE_GENERATION_PAYLOAD.scheduler
    CFG_SCALE = IMAGE_GENERATION_PAYLOAD.cfg_scale
    DISTILLED_CFG_SCALE = IMAGE_GENERATION_PAYLOAD.distilled_cfg_scale
    HR_CFG = IMAGE_GENERATION_PAYLOAD.hr_cfg
    HR_DISTILLED_CFG = IMAGE_GENERATION_PAYLOAD.hr_distilled_cfg
    WIDTH = IMAGE_GENERATION_PAYLOAD.width
    HEIGHT = IMAGE_GENERATION_PAYLOAD.height
    HR_UPSCALER = IMAGE_GENERATION_PAYLOAD.hr_upscaler
    DENOISING_STRENGTH = IMAGE_GENERATION_PAYLOAD.denoising_strength
    HR_SECOND_PASS_STEPS = IMAGE_GENERATION_PAYLOAD.hr_second_pass_steps
    HR_SCALE = IMAGE_GENERATION_PAYLOAD.hr_scale
    cl_logger.info(
        f"Image generation payload loaded: negative_prompt={NEGATIVE_PROMPT}, steps={STEPS}, sampler_name={SAMPLER_NAME}, scheduler={SCHEDULER}, cfg_scale={CFG_SCALE}, width={WIDTH}, height={HEIGHT}, hr_scale={HR_SCALE}, hr_upscaler={HR_UPSCALER}, denoising_strength={DENOISING_STRENGTH}, hr_second_pass_steps={HR_SECOND_PASS_STEPS}"
    )

    # Search enabled
    SEARCH_ENABLED = config.search_enabled

    # Timeouts
    IMAGE_GENERATION_TIMEOUT = config.timeouts.image_generation_timeout
    cl_logger.info(f"Image generation timeout loaded: {IMAGE_GENERATION_TIMEOUT}")

    # Token limits
    LLM_MAX_TOKENS = config.llm.max_tokens
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
    LLM_SETTINGS = config.llm_settings
    LLM_CHUNK_SIZE = LLM_SETTINGS.chunk_size
    LLM_TEMPERATURE = LLM_SETTINGS.temperature
    LLM_MODEL_NAME = LLM_SETTINGS.model_name
    cl_logger.info(
        f"LLM settings loaded: chunk_size={LLM_CHUNK_SIZE}, temperature={LLM_TEMPERATURE}, model_name={LLM_MODEL_NAME}"
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
    CHAINLIT_STARTERS = config.chainlit.starters
    cl_logger.info(f"Chainlit starters loaded: {CHAINLIT_STARTERS}")

    # Logging settings
    LOGGING_LEVEL = config.logging.level
    LOGGING_FILE = config.logging.file
    LOGGING_CONSOLE = config.logging.console
    LOGGING_FORMAT = config.logging.format
    LOGGING_MAX_SIZE = config.logging.max_size
    LOGGING_BACKUP_COUNT = config.logging.backup_count

    # Set up logging
    cl_logger.setLevel(LOGGING_LEVEL)
    formatter = logging.Formatter(LOGGING_FORMAT)
    handler = RotatingFileHandler(
        LOGGING_FILE, maxBytes=int(LOGGING_MAX_SIZE), backupCount=LOGGING_BACKUP_COUNT
    )
    handler.setFormatter(formatter)
    cl_logger.addHandler(handler)

    if LOGGING_CONSOLE:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        cl_logger.addHandler(console_handler)

    # Error handling settings
    MAX_RETRIES = config.error_handling.max_retries
    RETRY_DELAY = config.error_handling.retry_delay
    ERROR_LOG_LEVEL = config.error_handling.log_level

    # API settings
    API_BASE_URL = config.api.base_url
    API_TIMEOUT = config.api.timeout
    API_MAX_CONNECTIONS = config.api.max_connections

    # Feature toggles
    IMAGE_GENERATION_ENABLED = config.features.image_generation
    WEB_SEARCH_ENABLED = config.features.web_search
    DICE_ROLLING_ENABLED = config.features.dice_rolling

    # Rate limits
    IMAGE_GENERATION_RATE_LIMIT = config.rate_limits.image_generation
    API_CALLS_RATE_LIMIT = config.rate_limits.api_calls

    # Security settings
    SECRET_KEY = config.security.secret_key
    SESSION_TIMEOUT = config.security.session_timeout

    # Monitoring settings
    MONITORING_ENABLED = config.monitoring.enabled
    MONITORING_ENDPOINT = config.monitoring.endpoint
    MONITORING_SAMPLE_RATE = config.monitoring.sample_rate

    # Caching settings
    CACHING_ENABLED = config.caching.enabled
    CACHING_TTL = config.caching.ttl
    CACHING_MAX_SIZE = config.caching.max_size

    return config
