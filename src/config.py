import os
import yaml
import logging
from pydantic import BaseModel, ValidationError
from logging.handlers import RotatingFileHandler

# Initialize logging
cl_logger = logging.getLogger("chainlit")

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
    return config

# Load the config when the module is imported
config = load_config()

# Expose all required variables as module-level attributes
DICE_ROLLING_ENABLED = config.features.dice_rolling
DICE_SIDES = config.dice.sides
WEB_SEARCH_ENABLED = config.features.web_search
DATABASE_URL = os.getenv("DATABASE_URL", config.defaults.db_file)
KNOWLEDGE_DIRECTORY = config.paths.knowledge
LLM_MAX_TOKENS = config.llm.max_tokens
LLM_TEMPERATURE = config.llm.temperature
LLM_MODEL_NAME = config.llm.model_name
LLM_STREAMING = config.llm.streaming
LLM_TIMEOUT = config.llm.timeout
LLM_PRESENCE_PENALTY = config.llm.presence_penalty
LLM_FREQUENCY_PENALTY = config.llm.frequency_penalty
LLM_TOP_P = config.llm.top_p
LLM_VERBOSE = config.llm.verbose
IMAGE_GENERATION_ENABLED = config.features.image_generation
DECISION_AGENT_TEMPERATURE = config.agents.decision_agent.temperature
DECISION_AGENT_MAX_TOKENS = config.agents.decision_agent.max_tokens
DECISION_AGENT_STREAMING = config.agents.decision_agent.streaming
DECISION_AGENT_VERBOSE = config.agents.decision_agent.verbose
WRITER_AGENT_TEMPERATURE = config.agents.writer_agent.temperature
WRITER_AGENT_MAX_TOKENS = config.agents.writer_agent.max_tokens
WRITER_AGENT_STREAMING = config.agents.writer_agent.streaming
WRITER_AGENT_VERBOSE = config.agents.writer_agent.verbose
STORYBOARD_EDITOR_AGENT_TEMPERATURE = config.agents.storyboard_editor_agent.temperature
STORYBOARD_EDITOR_AGENT_MAX_TOKENS = config.agents.storyboard_editor_agent.max_tokens
STORYBOARD_EDITOR_AGENT_STREAMING = config.agents.storyboard_editor_agent.streaming
STORYBOARD_EDITOR_AGENT_VERBOSE = config.agents.storyboard_editor_agent.verbose
IMAGE_GENERATION_PAYLOAD = config.image_generation_payload
TIMEOUTS = config.timeouts
REFUSAL_LIST = config.refusal_list
DEFAULTS = config.defaults
PATHS = config.paths
OPENAI_SETTINGS = config.openai
SEARCH_SETTINGS = config.search
FEATURES = config.features
ERROR_HANDLING = config.error_handling
LOGGING = config.logging
API_SETTINGS = config.api
SECURITY_SETTINGS = config.security
MONITORING_SETTINGS = config.monitoring
CACHING_SETTINGS = config.caching
AGENTS = config.agents
CHAINLIT_SETTINGS = config.chainlit

# Configure logging using the loaded config
logging.basicConfig(
    level=LOGGING.level,
    format=LOGGING.format,
    handlers=[
        RotatingFileHandler(
            LOGGING.file,
            maxBytes=int(LOGGING.max_size),
            backupCount=LOGGING.backup_count
        ),
        logging.StreamHandler() if LOGGING.console else None
    ],
)

# Database configuration with fallbacks
cl_logger.info(f"Database URL loaded: {DATABASE_URL}")

# LLM configuration
cl_logger.info(
    f"LLM configuration loaded: "
    f"temperature={LLM_TEMPERATURE}, "
    f"max_tokens={LLM_MAX_TOKENS}, "
    f"model_name={LLM_MODEL_NAME}, "
    f"streaming={LLM_STREAMING}, "
    f"timeout={LLM_TIMEOUT}, "
    f"presence_penalty={LLM_PRESENCE_PENALTY}, "
    f"frequency_penalty={LLM_FREQUENCY_PENALTY}, "
    f"top_p={LLM_TOP_P}, "
    f"verbose={LLM_VERBOSE}"
)

# Agents configuration
cl_logger.info(
    f"Agents configuration loaded: "
    f"decision_agent (temperature={DECISION_AGENT_TEMPERATURE}, "
    f"max_tokens={DECISION_AGENT_MAX_TOKENS}, "
    f"streaming={DECISION_AGENT_STREAMING}, "
    f"verbose={DECISION_AGENT_VERBOSE}), "
    f"writer_agent (temperature={WRITER_AGENT_TEMPERATURE}, "
    f"max_tokens={WRITER_AGENT_MAX_TOKENS}, "
    f"streaming={WRITER_AGENT_STREAMING}, "
    f"verbose={WRITER_AGENT_VERBOSE}), "
    f"storyboard_editor_agent (temperature={STORYBOARD_EDITOR_AGENT_TEMPERATURE}, "
    f"max_tokens={STORYBOARD_EDITOR_AGENT_MAX_TOKENS}, "
    f"streaming={STORYBOARD_EDITOR_AGENT_STREAMING}, "
    f"verbose={STORYBOARD_EDITOR_AGENT_VERBOSE})"
)

# Image generation payload
cl_logger.info(
    f"Image generation payload loaded: "
    f"negative_prompt={IMAGE_GENERATION_PAYLOAD['negative_prompt']}, "
    f"steps={IMAGE_GENERATION_PAYLOAD['steps']}, "
    f"sampler_name={IMAGE_GENERATION_PAYLOAD['sampler_name']}, "
    f"scheduler={IMAGE_GENERATION_PAYLOAD['scheduler']}, "
    f"cfg_scale={IMAGE_GENERATION_PAYLOAD['cfg_scale']}, "
    f"width={IMAGE_GENERATION_PAYLOAD['width']}, "
    f"height={IMAGE_GENERATION_PAYLOAD['height']}, "
    f"hr_upscaler={IMAGE_GENERATION_PAYLOAD['hr_upscaler']}, "
    f"denoising_strength={IMAGE_GENERATION_PAYLOAD['denoising_strength']}, "
    f"hr_second_pass_steps={IMAGE_GENERATION_PAYLOAD['hr_second_pass_steps']}"
)

# Timeouts
cl_logger.info(f"Image generation timeout loaded: {TIMEOUTS['image_generation_timeout']}")

# Token limits
cl_logger.info(f"LLM max tokens loaded: {LLM_MAX_TOKENS}")

# Refusal list
cl_logger.info(f"Refusal list loaded: {REFUSAL_LIST}")

# Defaults
cl_logger.info(f"Default DB file loaded: {DEFAULTS['db_file']}")

# Dice settings
cl_logger.info(f"Default dice sides loaded: {DICE_SIDES}")

# Knowledge directory
cl_logger.info(f"Knowledge directory loaded: {KNOWLEDGE_DIRECTORY}")

# LLM settings
cl_logger.info(
    f"LLM settings loaded: "
    f"base_url={OPENAI_SETTINGS['base_url']}, "
    f"api_key={OPENAI_SETTINGS['api_key']}"
)

# Search settings
cl_logger.info(f"Search settings loaded: serpapi_key={SEARCH_SETTINGS['serpapi_key']}")

# Features
cl_logger.info(f"Features loaded: {FEATURES}")

# Error handling
cl_logger.info(f"Error handling settings loaded: {ERROR_HANDLING}")

# Logging
cl_logger.info(f"Logging settings loaded: {LOGGING}")

# API settings
cl_logger.info(f"API settings loaded: {API_SETTINGS}")

# Security settings
cl_logger.info(f"Security settings loaded: {SECURITY_SETTINGS}")

# Monitoring settings
cl_logger.info(f"Monitoring settings loaded: {MONITORING_SETTINGS}")

# Caching settings
cl_logger.info(f"Caching settings loaded: {CACHING_SETTINGS}")

# Agents settings
cl_logger.info(f"Agents settings loaded: {AGENTS}")

# Chainlit settings
cl_logger.info(f"Chainlit settings loaded: {CHAINLIT_SETTINGS}")
