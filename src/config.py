import os
import yaml
import logging

# Initialize logging
cl_logger = logging.getLogger("chainlit")

# Load configuration from YAML file
CONFIG_FILE = "config.yaml"
if not os.path.exists(CONFIG_FILE):
    cl_logger.error(f"Configuration file '{CONFIG_FILE}' not found.")
    raise FileNotFoundError(f"Configuration file '{CONFIG_FILE}' not found.")

with open(CONFIG_FILE, "r") as f:
    config_yaml = yaml.safe_load(f)

# Database configuration with fallbacks
DATABASE_URL = os.getenv("DATABASE_URL", config_yaml.get("defaults", {}).get("db_file", "chainlit.db"))
STABLE_DIFFUSION_API_URL = os.getenv("STABLE_DIFFUSION_API_URL", "http://localhost:7860")
cl_logger.info(f"Database URL loaded: {DATABASE_URL}")

# LLM configuration
LLM_TEMPERATURE = config_yaml.get("llm", {}).get("temperature", 0.7)
LLM_MAX_TOKENS = config_yaml.get("llm", {}).get("max_tokens", 64000)
LLM_MODEL_NAME = config_yaml.get("llm", {}).get("model_name", "gpt-3.5-turbo")
LLM_STREAMING = config_yaml.get("llm", {}).get("streaming", True)
LLM_TIMEOUT = config_yaml.get("llm", {}).get("timeout", 300)
LLM_PRESENCE_PENALTY = config_yaml.get("llm", {}).get("presence_penalty", 0.1)
LLM_FREQUENCY_PENALTY = config_yaml.get("llm", {}).get("frequency_penalty", 0.1)
LLM_TOP_P = config_yaml.get("llm", {}).get("top_p", 0.9)
LLM_VERBOSE = config_yaml.get("llm", {}).get("verbose", False)
cl_logger.info(f"LLM configuration loaded: temperature={LLM_TEMPERATURE}, max_tokens={LLM_MAX_TOKENS}, model_name={LLM_MODEL_NAME}, streaming={LLM_STREAMING}, timeout={LLM_TIMEOUT}, presence_penalty={LLM_PRESENCE_PENALTY}, frequency_penalty={LLM_FREQUENCY_PENALTY}, top_p={LLM_TOP_P}, verbose={LLM_VERBOSE}")

# Agents configuration
DECISION_AGENT_TEMPERATURE = config_yaml.get("agents", {}).get("decision_agent", {}).get("temperature", 0.2)
DECISION_AGENT_MAX_TOKENS = config_yaml.get("agents", {}).get("decision_agent", {}).get("max_tokens", 100)
DECISION_AGENT_STREAMING = config_yaml.get("agents", {}).get("decision_agent", {}).get("streaming", True)
DECISION_AGENT_VERBOSE = config_yaml.get("agents", {}).get("decision_agent", {}).get("verbose", True)
WRITER_AGENT_TEMPERATURE = config_yaml.get("agents", {}).get("writer_agent", {}).get("temperature", 0.7)
WRITER_AGENT_MAX_TOKENS = config_yaml.get("agents", {}).get("writer_agent", {}).get("max_tokens", 8000)
WRITER_AGENT_STREAMING = config_yaml.get("agents", {}).get("writer_agent", {}).get("streaming", True)
WRITER_AGENT_VERBOSE = config_yaml.get("agents", {}).get("writer_agent", {}).get("verbose", True)
STORYBOARD_EDITOR_AGENT_TEMPERATURE = config_yaml.get("agents", {}).get("storyboard_editor_agent", {}).get("temperature", 0.7)
STORYBOARD_EDITOR_AGENT_MAX_TOKENS = config_yaml.get("agents", {}).get("storyboard_editor_agent", {}).get("max_tokens", 8000)
STORYBOARD_EDITOR_AGENT_STREAMING = config_yaml.get("agents", {}).get("storyboard_editor_agent", {}).get("streaming", False)
STORYBOARD_EDITOR_AGENT_VERBOSE = config_yaml.get("agents", {}).get("storyboard_editor_agent", {}).get("verbose", True)
cl_logger.info(f"Agents configuration loaded: decision_agent (temperature={DECISION_AGENT_TEMPERATURE}, max_tokens={DECISION_AGENT_MAX_TOKENS}, streaming={DECISION_AGENT_STREAMING}, verbose={DECISION_AGENT_VERBOSE}), writer_agent (temperature={WRITER_AGENT_TEMPERATURE}, max_tokens={WRITER_AGENT_MAX_TOKENS}, streaming={WRITER_AGENT_STREAMING}, verbose={WRITER_AGENT_VERBOSE}), storyboard_editor_agent (temperature={STORYBOARD_EDITOR_AGENT_TEMPERATURE}, max_tokens={STORYBOARD_EDITOR_AGENT_MAX_TOKENS}, streaming={STORYBOARD_EDITOR_AGENT_STREAMING}, verbose={STORYBOARD_EDITOR_AGENT_VERBOSE})")

# Prompts
AI_WRITER_PROMPT = config_yaml.get("prompts", {}).get("ai_writer_prompt", "")
STORYBOARD_GENERATION_PROMPT = config_yaml.get("prompts", {}).get("storyboard_generation_prompt", "")
STORYBOARD_GENERATION_PROMPT_PREFIX = config_yaml.get("prompts", {}).get("storyboard_generation_prompt_prefix", "")
STORYBOARD_GENERATION_PROMPT_POSTFIX = config_yaml.get("prompts", {}).get("storyboard_generation_prompt_postfix", "")
DECISION_PROMPT = config_yaml.get("prompts", {}).get("decision_prompt", "")  # Load the new prompt

# Image generation payload
IMAGE_GENERATION_PAYLOAD = config_yaml.get("image_generation_payload", {})
NEGATIVE_PROMPT = IMAGE_GENERATION_PAYLOAD.get("negative_prompt", "")
STEPS = IMAGE_GENERATION_PAYLOAD.get("steps", 40)
SAMPLER_NAME = IMAGE_GENERATION_PAYLOAD.get("sampler_name", "Euler")
SCHEDULER = IMAGE_GENERATION_PAYLOAD.get("scheduler", "Simple")
CFG_SCALE = IMAGE_GENERATION_PAYLOAD.get("cfg_scale", 10)
DISTILLED_CFG_SCALE = IMAGE_GENERATION_PAYLOAD.get("distilled_cfg_scale", 1)
HR_CFG = IMAGE_GENERATION_PAYLOAD.get("hr_cfg", 1)
HR_DISTILLED_CFG = IMAGE_GENERATION_PAYLOAD.get("hr_distilled_cfg", 1)
WIDTH = IMAGE_GENERATION_PAYLOAD.get("width", 1024)
HEIGHT = IMAGE_GENERATION_PAYLOAD.get("height", 1024)
HR_UPSCALER = IMAGE_GENERATION_PAYLOAD.get("hr_upscaler", "RealESRGAN")
DENOISING_STRENGTH = IMAGE_GENERATION_PAYLOAD.get("denoising_strength", 0.1)
HR_SECOND_PASS_STEPS = IMAGE_GENERATION_PAYLOAD.get("hr_second_pass_steps", 0)
HR_SCALE = IMAGE_GENERATION_PAYLOAD.get("hr_scale", 1)
cl_logger.info(f"Image generation payload loaded: negative_prompt={NEGATIVE_PROMPT}, steps={STEPS}, sampler_name={SAMPLER_NAME}, scheduler={SCHEDULER}, cfg_scale={CFG_SCALE}, width={WIDTH}, height={HEIGHT}, hr_scale={HR_SCALE}, hr_upscaler={HR_UPSCALER}, denoising_strength={DENOISING_STRENGTH}, hr_second_pass_steps={HR_SECOND_PASS_STEPS}")

# Search enabled
SEARCH_ENABLED = config_yaml.get("search_enabled", False)

# Timeouts
IMAGE_GENERATION_TIMEOUT = config_yaml.get("timeouts", {}).get("image_generation_timeout", 300)
cl_logger.info(f"Image generation timeout loaded: {IMAGE_GENERATION_TIMEOUT}")

# Token limits
LLM_MAX_TOKENS = config_yaml.get("llm", {}).get("max_tokens", 4000)
cl_logger.info(f"LLM max tokens loaded: {LLM_MAX_TOKENS}")

# Refusal list
REFUSAL_LIST = config_yaml.get("refusal_list", [])
cl_logger.info(f"Refusal list loaded: {REFUSAL_LIST}")

# Defaults
DB_FILE = config_yaml.get("defaults", {}).get("db_file", "chainlit.db")
cl_logger.info(f"Default DB file loaded: {DB_FILE}")

# Dice settings
DICE_SIDES = config_yaml.get("dice", {}).get("sides", 20)  # Default to d20
cl_logger.info(f"Default dice sides loaded: {DICE_SIDES}")

# Knowledge directory
KNOWLEDGE_DIRECTORY = config_yaml.get("paths", {}).get("knowledge", "./knowledge")
cl_logger.info(f"Knowledge directory loaded: {KNOWLEDGE_DIRECTORY}")

# LLM settings
LLM_SETTINGS = config_yaml.get("llm_settings", {})
LLM_CHUNK_SIZE = LLM_SETTINGS.get("chunk_size", 1024)
LLM_TEMPERATURE = LLM_SETTINGS.get("temperature", 0.7)
LLM_MODEL_NAME = LLM_SETTINGS.get("model_name", "gpt-3.5-turbo")
cl_logger.info(f"LLM settings loaded: chunk_size={LLM_CHUNK_SIZE}, temperature={LLM_TEMPERATURE}, model_name={LLM_MODEL_NAME}")

# Image settings
IMAGE_SETTINGS = config_yaml.get("image_settings", {})
NUM_IMAGE_PROMPTS = IMAGE_SETTINGS.get("num_image_prompts", 1)
cl_logger.info(f"Image settings loaded: num_image_prompts={NUM_IMAGE_PROMPTS}")

# OpenAI settings
OPENAI_BASE_URL = config_yaml.get("openai", {}).get("base_url", "")
OPENAI_API_KEY = config_yaml.get("openai", {}).get("api_key", "")
cl_logger.info(f"OpenAI settings loaded: base_url={OPENAI_BASE_URL}, api_key={OPENAI_API_KEY}")

# Search settings
SERPAPI_KEY = config_yaml.get("search", {}).get("serpapi_key", "")
cl_logger.info(f"Search settings loaded: serpapi_key={SERPAPI_KEY}")

# Chainlit Starters
CHAINLIT_STARTERS = config_yaml.get("CHAINLIT_STARTERS", [
    "Hello! What kind of story would you like to create today?",
])
cl_logger.info(f"Chainlit starters loaded: {CHAINLIT_STARTERS}")

# Logging settings
LOGGING_LEVEL = config_yaml.get("logging", {}).get("level", "INFO")
LOGGING_FILE = config_yaml.get("logging", {}).get("file", "chainlit.log")
cl_logger.setLevel(LOGGING_LEVEL)
handler = logging.FileHandler(LOGGING_FILE)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
cl_logger.addHandler(handler)

# Error handling settings
MAX_RETRIES = config_yaml.get("error_handling", {}).get("max_retries", 3)
RETRY_DELAY = config_yaml.get("error_handling", {}).get("retry_delay", 5)
ERROR_LOG_LEVEL = config_yaml.get("error_handling", {}).get("log_level", "ERROR")

# API settings
API_BASE_URL = config_yaml.get("api", {}).get("base_url", "http://localhost:8080")
API_TIMEOUT = config_yaml.get("api", {}).get("timeout", 30)
API_MAX_CONNECTIONS = config_yaml.get("api", {}).get("max_connections", 10)

# Feature toggles
IMAGE_GENERATION_ENABLED = config_yaml.get("features", {}).get("image_generation", True)
WEB_SEARCH_ENABLED = config_yaml.get("features", {}).get("web_search", False)
DICE_ROLLING_ENABLED = config_yaml.get("features", {}).get("dice_rolling", True)

# Rate limits
IMAGE_GENERATION_RATE_LIMIT = config_yaml.get("rate_limits", {}).get("image_generation", "5/minute")
API_CALLS_RATE_LIMIT = config_yaml.get("rate_limits", {}).get("api_calls", "60/hour")

# Security settings
SECRET_KEY = config_yaml.get("security", {}).get("secret_key", "your-secret-key-here")
SESSION_TIMEOUT = config_yaml.get("security", {}).get("session_timeout", 3600)

# Monitoring settings
MONITORING_ENABLED = config_yaml.get("monitoring", {}).get("enabled", False)
MONITORING_ENDPOINT = config_yaml.get("monitoring", {}).get("endpoint", "http://localhost:9411")
MONITORING_SAMPLE_RATE = config_yaml.get("monitoring", {}).get("sample_rate", 0.1)

# Caching settings
CACHING_ENABLED = config_yaml.get("caching", {}).get("enabled", True)
CACHING_TTL = config_yaml.get("caching", {}).get("ttl", 3600)
CACHING_MAX_SIZE = config_yaml.get("caching", {}).get("max_size", "100MB")
