import os
import random
import requests
import re
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.core.agent import Agent
from langchain_core.llms import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)
from pydantic import BaseModel, Field, Literal  # Import Literal
from typing import List, Tuple, Optional  # Import List and Tuple
import logging
from .config import (
    DECISION_AGENT_TEMPERATURE,
    DECISION_AGENT_MAX_TOKENS,
    DECISION_AGENT_STREAMING,
    DECISION_AGENT_VERBOSE,
    WRITER_AGENT_TEMPERATURE,
    WRITER_AGENT_MAX_TOKENS,
    WRITER_AGENT_STREAMING,
    WRITER_AGENT_VERBOSE,
    STORYBOARD_EDITOR_AGENT_TEMPERATURE,
    STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
    STORYBOARD_EDITOR_AGENT_STREAMING,
    STORYBOARD_EDITOR_AGENT_VERBOSE,
    OPENAI_BASE_URL,
    SERPAPI_KEY,
    DICE_SIDES,
    WEB_SEARCH_ENABLED,
    DICE_ROLLING_ENABLED,
    LLM_PRESENCE_PENALTY,
    LLM_FREQUENCY_PENALTY,
    LLM_TOP_P,
    LLM_TIMEOUT,
    LLM_VERBOSE,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
)
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver

# Initialize logging
cl_logger = logging.getLogger("chainlit")


class DecisionOutput(BaseModel):
    """Schema for the decision output."""
    action: Literal["roll", "search", "continue_story"] = Field(
        description="The next action to take based on user input"
    )


from langgraph.prebuilt import tool  # Import tool from langgraph.prebuilt

from langgraph.prebuilt import create_react_agent
from langgraph.core.agent import Agent
from langchain_core.messages import ToolMessage

# Initialize the dice roll agent
dice_roll_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=0.0,
        max_tokens=100,
        streaming=False,
        verbose=False,
        request_timeout=LLM_TIMEOUT,
    ),
    tools=[],
    checkpointer=MemorySaver(),
)

@dice_roll_agent.tool
def dice_roll(input_str: Optional[str] = None) -> ToolMessage:
    """Roll dice based on user input.

    Args:
        input_str (str, optional): The dice specification (e.g., "d3", "2d6").
                                 Defaults to "d20" if not specified.

    Returns:
        ToolMessage: The result of the dice roll.
    """
    if not DICE_ROLLING_ENABLED:
        return ToolMessage(content="Dice rolling is disabled in the configuration.")

    try:
        # Default to d20 if no input is provided
        if not input_str:
            sides = DICE_SIDES
            count = 1
        else:
            # Parse the input to get dice specifications
            dice_list = parse_dice_input(input_str)
            if not dice_list:
                # Fallback to d20 if parsing fails
                sides = DICE_SIDES  # Corrected variable name
                count = 1
            else:
                # Use the first parsed dice specification
                sides, count = dice_list[0]

        # Roll the dice
        rolls = [random.randint(1, sides) for _ in range(count)]
        total = sum(rolls)

        # Format the result
        if count == 1:
            result = f"🎲 You rolled a {total} on a {sides}-sided die."
        else:
            result = f"🎲 You rolled {rolls} (total: {total}) on {count}d{sides}."

        cl_logger.info(f"Dice roll result: {result}")
        return ToolMessage(content=result)

    except ValueError as e:
        cl_logger.error(f"Dice roll failed: {e}", exc_info=True)
        return ToolMessage(content=f"🎲 Error rolling dice: {str(e)}")
    except Exception as e:
        cl_logger.error(f"Dice roll failed: {e}", exc_info=True)
        return ToolMessage(content=f"🎲 Error rolling dice: {str(e)}")


def parse_dice_input(input_str: str) -> List[Tuple[int, int]]:
    """Parse dice input string into a list of (sides, count) tuples."""
    pattern = r"(\d*)d(\d+)"
    matches = re.findall(pattern, input_str)
    dice_list = []

    for match in matches:
        count_str, sides_str = match
        try:
            count = int(count_str) if count_str else 1
            sides = int(sides_str)
            if sides < 1:
                raise ValueError("Invalid dice sides")
            dice_list.append((sides, count))
        except ValueError as e:
            cl_logger.error(f"Invalid dice specification: {e}")
            raise ValueError("Invalid dice specification") from e

    return dice_list


# Initialize the web search agent
web_search_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=0.0,
        max_tokens=100,
        streaming=False,
        verbose=False,
        request_timeout=LLM_TIMEOUT,
    ),
    tools=[],
    checkpointer=MemorySaver(),
)

@web_search_agent.tool
def web_search(query: str) -> ToolMessage:
    """Perform a web search using SerpAPI.

    Args:
        query (str): The search query.

    Returns:
        ToolMessage: The search result.
    """
    if not SERPAPI_KEY:
        return ToolMessage(content="SERPAPI_KEY environment variable not set.")
    if not WEB_SEARCH_ENABLED:
        return ToolMessage(content="Web search is disabled.")
    url = f"https://serpapi.com/search.json?q={query}&api_key={SERPAPI_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError(f"Search error: {data['error']}")
        return ToolMessage(content=data.get("organic_results", [{}])[0].get("snippet", "No results found."))
    except requests.exceptions.RequestException as e:
        cl_logger.error(f"Web search failed: {e}", exc_info=True)
        return ToolMessage(content=f"Web search failed: {str(e)}")
    except ValueError as e:
        cl_logger.error(f"Web search failed: {e}", exc_info=True)
        return ToolMessage(content=f"Web search failed: {str(e)}")


# Initialize the decision agent
decision_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=DECISION_AGENT_TEMPERATURE,
        max_tokens=DECISION_AGENT_MAX_TOKENS,
        streaming=DECISION_AGENT_STREAMING,
        verbose=DECISION_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 2,
    ),
    tools=[dice_roll_agent, web_search_agent],
    checkpointer=MemorySaver(),
)

# Initialize the writer AI agent
writer_agent = Agent(
    model=ChatOpenAI(
        temperature=WRITER_AGENT_TEMPERATURE,
        max_tokens=WRITER_AGENT_MAX_TOKENS,
        streaming=WRITER_AGENT_STREAMING,
        verbose=WRITER_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 3,
        presence_penalty=LLM_PRESENCE_PENALTY,
        frequency_penalty=LLM_FREQUENCY_PENALTY,
        top_p=LLM_TOP_P,
    ),
    tools=[dice_roll, web_search],
    checkpointer=MemorySaver(),
)

# Initialize the storyboard editor agent
storyboard_editor_agent = Agent(
    model=ChatOpenAI(
        temperature=STORYBOARD_EDITOR_AGENT_TEMPERATURE,
        max_tokens=STORYBOARD_EDITOR_AGENT_MAX_TOKENS,
        streaming=STORYBOARD_EDITOR_AGENT_STREAMING,
        verbose=STORYBOARD_EDITOR_AGENT_VERBOSE,
        request_timeout=LLM_TIMEOUT * 2,
    ),
    tools=[dice_roll, web_search],
    checkpointer=MemorySaver(),
)
