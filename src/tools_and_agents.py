import random
from typing import Optional

from langgraph.prebuilt import ToolNode, ToolExecutor

from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI

from langchain.schema.output_parser import StrOutputParser
from langchain_core.tools import tool

from config import (
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_CHUNK_SIZE,
    LLM_MODEL_NAME,
    LLM_STREAMING,
    LLM_TIMEOUT,
    LLM_PRESENCE_PENALTY,
    LLM_FREQUENCY_PENALTY,
    LLM_TOP_P,
    LLM_VERBOSE,
    AI_WRITER_PROMPT,
    STORYBOARD_GENERATION_PROMPT,
    DICE_SIDES,
)

import logging


# Initialize logging
cl_logger = logging.getLogger("chainlit")

@tool
def dice_roll(n: Optional[int] = DICE_SIDES) -> str:
    """
    Rolls a dice with a specified number of sides.

    Args:
        n (Optional[int]): Number of sides on the dice. Defaults to DICE_SIDES.

    Returns:
        str: Result of the dice roll.
    """
    sides = n if n else DICE_SIDES
    result = random.randint(1, sides)
    cl_logger.info(f"Dice roll result: {result} on a {sides}-sided die.")
    return f"🎲 You rolled a {result} on a {sides}-sided die."

tools = [dice_roll]

# Initialize the writer AI agent
writer_model = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    temperature=LLM_TEMPERATURE,
    streaming=LLM_STREAMING,
    model_name=LLM_MODEL_NAME,
    request_timeout=LLM_TIMEOUT,
    max_tokens=LLM_MAX_TOKENS,
    presence_penalty=LLM_PRESENCE_PENALTY,
    frequency_penalty=LLM_FREQUENCY_PENALTY,
    top_p=LLM_TOP_P,
    verbose=LLM_VERBOSE
)

writer_prompt = PromptTemplate.from_template(AI_WRITER_PROMPT)

# Assuming ChatOpenAI has a bind_tools method; if not, adjust accordingly
writer_agent = writer_model.bind_tools(tools)

# Initialize the image generation AI agent
storyboard_model = ChatOpenAI(
    base_url="http://192.168.1.111:5000/v1",
    temperature=LLM_TEMPERATURE,
    streaming=False,
    model_name=LLM_MODEL_NAME,
    request_timeout=LLM_TIMEOUT,
    max_tokens=LLM_MAX_TOKENS,
    presence_penalty=LLM_PRESENCE_PENALTY,
    frequency_penalty=LLM_FREQUENCY_PENALTY,
    top_p=LLM_TOP_P,
    verbose=LLM_VERBOSE
)

storyboard_generation_agent = storyboard_model

storyboard_editor_agent = storyboard_model
