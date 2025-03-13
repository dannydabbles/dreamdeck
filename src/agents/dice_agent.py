from src.config import config
import os
import random
import logging
import re
from typing import Dict, List, Tuple, Optional
from json import loads
from uuid import uuid4  # Import uuid4
from langgraph.prebuilt import create_react_agent
from langgraph.func import task
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage  # Use LangChain's standard messages
from chainlit import Message as CLMessage  # Import CLMessage from Chainlit
from ..config import DICE_ROLLING_ENABLED, DICE_SIDES
from langchain_openai import ChatOpenAI  # Import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver  # Import MemorySaver
from ..models import ChatState

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def _dice_roll(state: ChatState) -> List[BaseMessage]:
    """Process dice rolling requests from users."""
    input_str = next((m for m in reversed(state.messages) if isinstance(m, HumanMessage)), None).content
    recent_chat = state.get_recent_history_str(n=5)  # Fetch last 5 messages
    
    try:
        # New LLM prompt construction
        formatted_prompt = config.prompts.dice_processing_prompt.format(
            user_query=input_str,
            recent_chat=recent_chat
        )
        
        # Invoke LLM to get structured output
        llm = ChatOpenAI(
            base_url=config.openai.base_url,
            temperature=0.7,  # Adjust temperature as needed
            max_tokens=100,
            streaming=False,
            verbose=True,
            timeout=config.llm.timeout
        )
        response = llm.invoke([('system', formatted_prompt)])
        json_output = loads(response.content.strip())

        # Validate and parse JSON response
        specs = json_output.get('specs', [])
        reasons = json_output.get('reasons', [])

        if len(specs) != len(reasons):
            raise ValueError("Mismatched specs/reasons lengths")

        # Perform actual dice rolls
        results = []
        for i, spec in enumerate(specs):
            parts = spec.split('d')
            count = int(parts[0] or '1')
            sides = int(parts[1])
            rolls = [random.randint(1, sides) for _ in range(count)]
            total = sum(rolls)
            
            results.append({
                'spec': spec,
                'outcome': f"{rolls} ({sum(rolls)})",
                'reason': reasons[i]
            })

        # Prepare messages
        lang_graph_msg = "\n".join([
            f"- {res['spec']} → {res['outcome']} ({res['reason']})"
            for res in results
        ])

        # Send ChainLit message
        cl_msg = CLMessage(
            content=f"**Dice Rolls:**\n\n" + "\n\n".join([
                f"• **{res['spec']}** → {res['outcome']} ({res['reason']})"
                for res in results
            ]),
            parent_id=None  # Attach to current thread
        )
        await cl_msg.send()

        return [
            AIMessage(
                content=lang_graph_msg,
                name="dice_roll"
            )
        ]

    except Exception as e:
        cl_logger.error(f"Dice roll failed: {e}")
        return [AIMessage(
            content=f"🎲 Error rolling dice: {str(e)}",
            name="error",
        )]

@task
async def dice_roll(state: ChatState) -> list[BaseMessage]:
    return await _dice_roll(state)

# Export the function as dice_roll_agent
dice_roll_agent = dice_roll

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
            if count < 1 or sides < 2:
                raise ValueError("Invalid dice specification")
            dice_list.append((sides, count))
        except ValueError as e:
            cl_logger.error(f"Invalid dice specification: {e}")
            # Gracefully skip invalid entries instead of raising
            continue  # Skip bad entries
    
    return dice_list

dice_agent = dice_roll
