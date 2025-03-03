import os
import random
import logging
from langgraph.prebuilt import create_react_agent
from langgraph.message import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from .config import DICE_ROLLING_ENABLED, DICE_SIDES

# Initialize logging
cl_logger = logging.getLogger("chainlit")

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
                sides = DICE_SIDES
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

# Initialize the dice roll agent
dice_roll_agent = create_react_agent(
    model=ChatOpenAI(
        temperature=0.0,
        max_tokens=100,
        streaming=False,
        verbose=False,
        request_timeout=os.getenv("LLM_TIMEOUT", 10),
    ),
    tools=[dice_roll],
    checkpointer=MemorySaver(),
)
