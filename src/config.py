import os
from pathlib import Path
import yaml
import logging
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from typing import ClassVar, Optional
from logging.handlers import RotatingFileHandler
import warnings  # Add this import

warnings.filterwarnings("ignore", category=DeprecationWarning)  # Add this line

# Initialize logging
cl_logger = logging.getLogger("chainlit")
cl_logger.setLevel(logging.DEBUG)

# Locate config.yaml relative to the package root
PROMPTS_DIR = Path(__file__).parent / "prompts"
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"
if not os.path.exists(CONFIG_FILE):
    cl_logger.error(f"Configuration file '{CONFIG_FILE}' not found.")
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

with open(CONFIG_FILE, "r") as f:
    config_yaml = yaml.safe_load(f)


class DiceConfig(BaseModel):
    """Pydantic model for dice configuration."""
    model_config = ConfigDict(extra="forbid")
    sides: int = Field(default=20, description="Number of sides for dice rolls")


class DecisionAgentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    temperature: float
    max_tokens: int
    streaming: bool
    base_url: Optional[str] = None
    verbose: bool
    tools: dict = Field(default_factory=dict)


class WriterAgentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    temperature: float
    max_tokens: int
    base_url: Optional[str] = None
    streaming: bool
    verbose: bool
    personas: dict = Field(default_factory=dict)


class StoryboardEditorAgentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float
    max_tokens: int
    base_url: Optional[str] = None
    streaming: bool
    verbose: bool


class AgentsConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    decision_agent: DecisionAgentConfig
    writer_agent: WriterAgentConfig
    storyboard_editor_agent: StoryboardEditorAgentConfig
    todo_agent: dict = Field(default_factory=dict)
    knowledge_agent: dict = Field(default_factory=dict)


class DefaultsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    db_file: str = "chainlit.db"


class LlmConfig(BaseModel):
    model: str
    model_config = ConfigDict(extra="forbid")
    temperature: float
    max_tokens: int
    model_name: str
    streaming: bool
    timeout: int
    presence_penalty: float
    frequency_penalty: float
    top_p: float
    verbose: bool


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dice_rolling: bool
    web_search: bool
    image_generation: bool


class ConfigSchema(BaseModel):
    """Central configuration container for the application."""

    llm: LlmConfig
    prompt_files: dict = Field(alias="prompts")
    loaded_prompts: dict = {}
    image_generation_payload: dict
    timeouts: dict
    refusal_list: list
    defaults: DefaultsConfig
    dice: DiceConfig
    paths: dict
    openai: dict
    search: dict
    features: FeatureConfig
    error_handling: dict
    logging: dict
    api: dict
    security: dict
    monitoring: dict
    caching: dict
    agents: AgentsConfig
    chainlit: dict
    search_enabled: bool
    knowledge_directory: str
    image_settings: dict
    rate_limits: dict
    storyboard_generation_prompt_prefix: str = ""
    storyboard_generation_prompt_postfix: str = ""
    todo_dir_path: str = "./helper"
    todo_file_name: str = "todo.md"
    max_chain_length: int = 3
    persona_tool_preferences: dict = Field(default_factory=dict)
    _env_prefix: ClassVar[str] = "APP_"
    model_config = ConfigDict(extra="forbid")
    chat: dict


def parse_size(size_str: str) -> int:
    """Parse size strings like '10MB' to bytes."""
    units = {"KB": 1024, "MB": 1024**2, "GB": 1024**3}
    size = size_str.upper()
    for suffix in units:
        if size.endswith(suffix):
            num = float(size[: -len(suffix)])
            return int(num * units[suffix])
    raise ValueError(f"Invalid size format: {size_str}")


def load_config():
    """Load and validate configuration from YAML file.

    Returns:
        ConfigSchema: Validated configuration object
    """
    schema = ConfigSchema.model_validate(config_yaml)

    # Load prompts from files
    for key, filename in schema.prompt_files.items():
        prompt_path = PROMPTS_DIR / filename
        if prompt_path.exists():
            schema.loaded_prompts[key] = prompt_path.read_text()
        else:
            cl_logger.error(f"Prompt file not found: {prompt_path}")
            schema.loaded_prompts[key] = f"ERROR: Prompt file {filename} not found."

    return schema


config = load_config()

PERSONA_TOOL_PREFERENCES = config.persona_tool_preferences

MAX_CHAIN_LENGTH = config.max_chain_length

DICE_ROLLING_ENABLED = config.features.dice_rolling
DICE_SIDES = config.dice.sides
WEB_SEARCH_ENABLED = config.features.web_search
DATABASE_URL = os.getenv("DATABASE_URL", config.defaults.db_file)
KNOWLEDGE_DIRECTORY = os.path.realpath(config.paths.get("knowledge", "./knowledge"))
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

WEB_SEARCH_PROMPT = config.loaded_prompts.get("web_search_prompt", "")

DECISION_AGENT_TEMPERATURE = config.agents.decision_agent.temperature
DECISION_AGENT_MAX_TOKENS = config.agents.decision_agent.max_tokens
DECISION_AGENT_STREAMING = config.agents.decision_agent.streaming
DECISION_AGENT_VERBOSE = config.agents.decision_agent.verbose
DECISION_AGENT_BASE_URL = config.agents.decision_agent.base_url

WRITER_AGENT_TEMPERATURE = config.agents.writer_agent.temperature
WRITER_AGENT_MAX_TOKENS = config.agents.writer_agent.max_tokens
WRITER_AGENT_STREAMING = config.agents.writer_agent.streaming
WRITER_AGENT_VERBOSE = config.agents.writer_agent.verbose
WRITER_AGENT_BASE_URL = config.agents.writer_agent.base_url

STORYBOARD_EDITOR_AGENT_TEMPERATURE = config.agents.storyboard_editor_agent.temperature
STORYBOARD_EDITOR_AGENT_MAX_TOKENS = config.agents.storyboard_editor_agent.max_tokens
STORYBOARD_EDITOR_AGENT_STREAMING = config.agents.storyboard_editor_agent.streaming
STORYBOARD_EDITOR_AGENT_VERBOSE = config.agents.storyboard_editor_agent.verbose
STORYBOARD_EDITOR_AGENT_BASE_URL = config.agents.storyboard_editor_agent.base_url

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
START_MESSAGE = config.chat.get("start_message", "Hello! How can I help you today?")

AI_WRITER_PROMPT = config.loaded_prompts.get("ai_writer_prompt", "Write a story.")
STORYBOARD_GENERATION_PROMPT = config.loaded_prompts.get(
    "storyboard_generation_prompt", "Generate a storyboard."
)

DIRECTOR_PROMPT = config.loaded_prompts.get(
    "director_prompt", "You are an AI director."
)

SERPAPI_KEY = os.getenv("SERPAPI_KEY", config.search.get("serpapi_key", ""))

handlers = [
    RotatingFileHandler(
        LOGGING["file"],
        maxBytes=parse_size(LOGGING["max_size"]),
        backupCount=LOGGING["backup_count"],
    ),
]
if LOGGING["console"]:
    handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=LOGGING["level"],
    format=LOGGING["format"],
    handlers=handlers,
)

DATABASE_URL = os.getenv("DATABASE_URL", config.defaults.db_file)
cl_logger.info(f"Database URL loaded: {DATABASE_URL}")

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

IMAGE_GENERATION_TIMEOUT = TIMEOUTS.get("image_generation_timeout", 180)
cl_logger.info(f"Image generation timeout loaded: {IMAGE_GENERATION_TIMEOUT}")

cl_logger.info(f"LLM max tokens loaded: {LLM_MAX_TOKENS}")

cl_logger.info(f"Refusal list loaded: {REFUSAL_LIST}")

cl_logger.info(f"Default DB file loaded: {config.defaults.db_file}")

cl_logger.info(f"Default dice sides loaded: {DICE_SIDES}")

cl_logger.info(f"Knowledge directory loaded: {KNOWLEDGE_DIRECTORY}")

cl_logger.info(
    f"LLM settings loaded: "
    f"base_url={OPENAI_SETTINGS['base_url']}, "
    f"api_key={OPENAI_SETTINGS['api_key']}"
)

STABLE_DIFFUSION_API_URL = config.image_generation_payload.get(
    "url", os.environ.get("STABLE_DIFFUSION_API_URL")
)

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

STORYBOARD_GENERATION_PROMPT_PREFIX = config.storyboard_generation_prompt_prefix
STORYBOARD_GENERATION_PROMPT_POSTFIX = config.storyboard_generation_prompt_postfix
TODO_DIR_PATH = config.todo_dir_path
TODO_FILE_NAME = config.todo_file_name

cl_logger.info(f"Search settings loaded: serpapi_key={SERPAPI_KEY}")

cl_logger.info(f"Features loaded: {FEATURES}")

cl_logger.info(f"Error handling settings loaded: {ERROR_HANDLING}")

cl_logger.info(f"Logging settings loaded: {LOGGING}")

cl_logger.info(f"API settings loaded: {API_SETTINGS}")

cl_logger.info(f"Security settings loaded: {SECURITY_SETTINGS}")

cl_logger.info(f"Monitoring settings loaded: {MONITORING_SETTINGS}")

cl_logger.info(f"Caching settings loaded: {CACHING_SETTINGS}")

cl_logger.info(f"Agents settings loaded: {AGENTS}")

cl_logger.info(f"Chainlit settings loaded: {CHAINLIT_SETTINGS}")
