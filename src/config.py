import os
from pathlib import Path
import yaml
import logging
from pydantic import BaseModel, Field, ValidationError
from logging.handlers import RotatingFileHandler

# Initialize logging
cl_logger = logging.getLogger("chainlit")

# Locate config.yaml relative to the package root
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
if not os.path.exists(CONFIG_FILE):
    cl_logger.error(f"Configuration file '{CONFIG_FILE}' not found.")
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

with open(CONFIG_FILE, "r") as f:
    config_yaml = yaml.safe_load(f)

class DiceConfig(BaseModel):
    """Pydantic model for dice configuration."""
    sides: int = Field(default=20, description="Number of sides for dice rolls")

from typing import ClassVar  # Move ClassVar to typing
from pydantic import ConfigDict  # Keep ConfigDict from pydantic

class LlmConfig(BaseModel):
    temperature: float
    max_tokens: int
    model_name: str
    streaming: bool
    timeout: int
    presence_penalty: float
    frequency_penalty: float
    top_p: float
    verbose: bool

class ConfigSchema(BaseModel):
    """Pydantic model for the configuration schema."""
    llm: LlmConfig
    prompts: dict
    image_generation_payload: dict
    timeouts: dict
    refusal_list: list
    defaults: dict
    dice: DiceConfig  # Use the nested DiceConfig model
    paths: dict
    openai: dict
    search: dict
    features: dict  # Add agents configuration
    error_handling: dict
    logging: dict
    api: dict
    security: dict
    monitoring: dict
    caching: dict
    agents: dict  # Add agents configuration
    chainlit: dict  # Add chainlit configuration
    search_enabled: bool  # Added
    knowledge_directory: str  # Added
    image_settings: dict  # Added
    rate_limits: dict  # Added
    storyboard_generation_prompt_prefix: str = ""  # Add this line
    storyboard_generation_prompt_postfix: str = ""  # Add this line
    _env_prefix: ClassVar[str] = "APP_"  # NEW: Enable env var loading with prefix
    model_config = ConfigDict(extra='forbid')  # ENFORCE strict validation

def parse_size(size_str: str) -> int:
    """Parse size strings like '10MB' to bytes."""
    units = {'KB': 1024, 'MB': 1024**2, 'GB': 1024**3}
    size = size_str.upper()
    for suffix in units:
        if size.endswith(suffix):
            num = float(size[:-len(suffix)])
            return int(num * units[suffix])
    raise ValueError(f"Invalid size format: {size_str}")

def load_config() -> ConfigSchema:
    """Load and validate configuration from YAML file.

    Returns:
        ConfigSchema: Validated configuration object
    """
    try:
        config_data = ConfigSchema.model_validate(config_yaml)  # USE VALIDATOR
    except ValidationError as e:
        cl_logger.error(f"Validation failed: {e.errors()}")  # BETTER ERROR REPORTING
        raise
    return config_data  # Ensure returns the validated model

# Load the config when the module is imported
config = load_config()

# Expose all required variables as module-level attributes
DICE_ROLLING_ENABLED = config.features['dice_rolling']
DICE_SIDES = config.dice.sides  # Now dice is a DiceConfig instance
WEB_SEARCH_ENABLED = config.features['web_search']
DATABASE_URL = os.getenv("DATABASE_URL", config.defaults['db_file'])
KNOWLEDGE_DIRECTORY = config.paths['knowledge']
LLM_MAX_TOKENS = config.llm.max_tokens
LLM_TEMPERATURE = config.llm.temperature
LLM_MODEL_NAME = config.llm.model_name
LLM_STREAMING = config.llm.streaming
LLM_TIMEOUT = config.llm.timeout
LLM_PRESENCE_PENALTY = config.llm.presence_penalty
LLM_FREQUENCY_PENALTY = config.llm.frequency_penalty
LLM_TOP_P = config.llm.top_p
LLM_VERBOSE = config.llm.verbose
IMAGE_GENERATION_ENABLED = config.features['image_generation']
DECISION_PROMPT=config.prompts['decision_prompt']
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

# Ensure SERPAPI_KEY is set from environment variable first, then config file
SERPAPI_KEY = os.getenv('SERPAPI_KEY', config.search.get('serpapi_key', ''))

# Configure logging using the loaded config
logging.basicConfig(
    level=LOGGING['level'],
    format=LOGGING['format'],
    handlers=[
        RotatingFileHandler(
            LOGGING['file'],
            maxBytes=parse_size(LOGGING['max_size']),  # Use parse_size
            backupCount=LOGGING['backup_count']
        ),
        logging.StreamHandler() if LOGGING['console'] else None
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
IMAGE_GENERATION_TIMEOUT = TIMEOUTS.get("image_generation_timeout", 180)  # Default from config.yaml
cl_logger.info(f"Image generation timeout loaded: {IMAGE_GENERATION_TIMEOUT}")

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

# Expose Stable Diffusion API URL
STABLE_DIFFUSION_API_URL = config.image_generation_payload.get("url", "https://example.com/sdapi")

CFG_SCALE = config.image_generation_payload.get("cfg_scale", 3.5)
DENOISING_STRENGTH = config.image_generation_payload.get("denoising_strength", 0.6)
STEPS = config.image_generation_payload.get("steps", 30)
SAMPLER_NAME = config.image_generation_payload.get("sampler_name", "Euler")
SCHEDULER = config.image_generation_payload.get("scheduler", "Simple")
WIDTH = config.image_generation_payload.get("width", 512)
HEIGHT = config.image_generation_payload.get("height", 512)
HR_UPSCALER = config.image_generation_payload.get("hr_upscaler", "SwinIR 4x")
HR_SECOND_PASS_STEPS = config.image_generation_payload.get("hr_second_pass_steps", 10)
NEGATIVE_PROMPT = config.image_generation_payload.get("negative_prompt", "")

# Expose storyboard generation prompt settings
STORYBOARD_GENERATION_PROMPT_PREFIX = config.storyboard_generation_prompt_prefix
STORYBOARD_GENERATION_PROMPT_POSTFIX = config.storyboard_generation_prompt_postfix

# Search settings
cl_logger.info(f"Search settings loaded: serpapi_key={SERPAPI_KEY}")

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
