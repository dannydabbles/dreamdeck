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

# Database configuration
DATABASE_URL = config_yaml.get("database", {}).get("url", "sqlite+aiosqlite:///chainlit.db")
cl_logger.info(f"Database URL loaded: {DATABASE_URL}")

# LLM configuration
LLM_TEMPERATURE = config_yaml.get("llm", {}).get("temperature", 0.7)
LLM_MAX_TOKENS = config_yaml.get("llm", {}).get("max_tokens", 64000)
LLM_CHUNK_SIZE = config_yaml.get("llm", {}).get("chunk_size", 1024)
LLM_MODEL_NAME = config_yaml.get("llm", {}).get("model_name", "gpt-3.5-turbo")
LLM_STREAMING = bool(config_yaml.get("llm", {}).get("streaming", True))
LLM_TIMEOUT = config_yaml.get("llm", {}).get("timeout", 300)
LLM_PRESENCE_PENALTY = config_yaml.get("llm", {}).get("presence_penalty", 0.1)
LLM_FREQUENCY_PENALTY = config_yaml.get("llm", {}).get("frequency_penalty", 0.1)
LLM_TOP_P = config_yaml.get("llm", {}).get("top_p", 0.9)
LLM_VERBOSE = config_yaml.get("llm", {}).get("verbose", False)
cl_logger.info(f"LLM configuration loaded: temperature={LLM_TEMPERATURE}, max_tokens={LLM_MAX_TOKENS}, chunk_size={LLM_CHUNK_SIZE}, model_name={LLM_MODEL_NAME}, streaming={LLM_STREAMING}, timeout={LLM_TIMEOUT}, presence_penalty={LLM_PRESENCE_PENALTY}, frequency_penalty={LLM_FREQUENCY_PENALTY}, top_p={LLM_TOP_P}, verbose={LLM_VERBOSE}")

# Prompts
AI_WRITER_PROMPT = config_yaml.get("prompts", {}).get("ai_writer_prompt", "")
STORYBOARD_GENERATION_PROMPT = config_yaml.get("prompts", {}).get("storyboard_generation_prompt", "")
STORYBOARD_GENERATION_PROMPT_PREFIX = config_yaml.get("prompts", {}).get("storyboard_generation_prompt_prefix", "")
STORYBOARD_GENERATION_PROMPT_POSTFIX = config_yaml.get("prompts", {}).get("storyboard_generation_prompt_postfix", "")

# Image generation payload
IMAGE_GENERATION_PAYLOAD = config_yaml.get("image_generation_payload", {})
NEGATIVE_PROMPT = IMAGE_GENERATION_PAYLOAD.get("negative_prompt", "")
STEPS = IMAGE_GENERATION_PAYLOAD.get("steps", 40)
SAMPLER_NAME = IMAGE_GENERATION_PAYLOAD.get("sampler_name", "Euler")
SCHEDULER = IMAGE_GENERATION_PAYLOAD.get("scheduler", "Simple")
CFG_SCALE = IMAGE_GENERATION_PAYLOAD.get("cfg_scale", 10)
WIDTH = IMAGE_GENERATION_PAYLOAD.get("width", 1024)
HEIGHT = IMAGE_GENERATION_PAYLOAD.get("height", 1024)
cl_logger.info(f"Image generation payload loaded: negative_prompt={NEGATIVE_PROMPT}, steps={STEPS}, sampler_name={SAMPLER_NAME}, scheduler={SCHEDULER}, cfg_scale={CFG_SCALE}, width={WIDTH}, height={HEIGHT}")

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
DICE_SIDES = config_yaml.get("dice", {}).get("sides", 100)
cl_logger.info(f"Dice sides loaded: {DICE_SIDES}")

# Knowledge directory
KNOWLEDGE_DIRECTORY = config_yaml.get("knowledge_directory", "./knowledge")
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

# Stable Diffusion API URL
STABLE_DIFFUSION_API_URL = config_yaml.get("stable_diffusion_api_url", "http://localhost:7860")
cl_logger.info(f"Stable Diffusion API URL loaded: {STABLE_DIFFUSION_API_URL}")

# Chainlit Starters
CHAINLIT_STARTERS = config_yaml.get("CHAINLIT_STARTERS", [
    "Hello! What kind of story would you like to create today?",
])
cl_logger.info(f"Chainlit starters loaded: {CHAINLIT_STARTERS}")
