import asyncio
import logging
import json  # Import json
from typing import List, Optional, Dict, Any
from langgraph.func import entrypoint, task
from langgraph.types import Command, StreamWriter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage  # Import SystemMessage
from chainlit import Message as CLMessage
from .state import ChatState  # Update import path
from .tools_and_agents import (
    writer_agent,
    storyboard_editor_agent,
    dice_roll,
    web_search,
    decision_agent,
    log_decision_agent_response
)
from .image_generation import process_storyboard_images, generate_image_generation_prompts
from .config import (
    DECISION_PROMPT,
    SEARCH_ENABLED,
    LLM_TIMEOUT,
    REFUSAL_LIST,
    DICE_SIDES,
    CHAINLIT_STARTERS,
    IMAGE_GENERATION_ENABLED,
    DICE_ROLLING_ENABLED,  # Add this import
    WEB_SEARCH_ENABLED  # Add this import
)
from .models import ChatState  # Update import path
from .memory_management import save_chat_memory  # Import save_chat_memory
from .tools_and_agents import generate_story_response  # Import generate_story_response

# Initialize logging
cl_logger = logging.getLogger("chainlit")

@task
async def process_storyboard(state: ChatState) -> Optional[str]:
    """Generate storyboard prompts based on the GM's last response."""
    try:
        # Get the last GM response
        last_gm_message = next((msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)), None)
        if not last_gm_message:
            cl_logger.warning("No GM message found to generate storyboard from")
            return None

        # Format the prompt with proper context
        formatted_prompt = DECISION_PROMPT.format(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str()
        )

        # Create messages list for the storyboard generation
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=last_gm_message.content)
        ]

        # Get storyboard prompts from the agent
        response = await storyboard_editor_agent.ainvoke(messages)

        if not response or not response.content:
            cl_logger.warning("Empty storyboard response")
            return None

        # Clean up the response - remove any thinking tags, etc.
        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()

        return content.strip()
    except Exception as e:
        cl_logger.error(f"Storyboard generation failed: {str(e)}", exc_info=True)
        return None

@entrypoint(checkpointer=MemorySaver())
async def chat_workflow(
    messages: List[BaseMessage],
    store: BaseStore,
    previous: Optional[ChatState] = None,
    writer: StreamWriter = None
) -> ChatState:
    """Main chat workflow handling messages and state."""
    state = previous or ChatState()
    state.messages.extend(messages)

    try:
        if writer:
            writer("Processing started")

        # Determine action
        last_human_message = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
        if not last_human_message:
            cl_logger.info("No human message found, defaulting to writer")
            action = "writer"
        else:
            formatted_prompt = DECISION_PROMPT.format(user_input=last_human_message.content)
            messages = [SystemMessage(content=formatted_prompt)]
            response = await decision_agent.ainvoke(messages)
            log_decision_agent_response(response)

            if (
                hasattr(response, 'additional_kwargs') 
                and 'function_call' in response.additional_kwargs
            ):
                function_call = response.additional_kwargs['function_call']
                try:
                    args = json.loads(function_call['arguments'])
                    action = args.get('action')
                    if (
                        "roll" in last_human_message.content.lower() or 
                        "d20" in last_human_message.content.lower() or
                        "dice" in last_human_message.content.lower()
                    ):
                        action = "roll"
                except json.JSONDecodeError as e:
                    cl_logger.error(f"Failed to parse function arguments: {e}")
                    action = 'continue_story'
            else:
                action = 'continue_story'

        action_map = {
            "roll": "roll",
            "search": "search" if SEARCH_ENABLED else "writer",
            "continue_story": "writer"
        }
        mapped_action = action_map.get(action, "writer")

        if mapped_action == "roll" and DICE_ROLLING_ENABLED:
            result = await dice_roll.ainvoke({"n": DICE_SIDES})
            state.messages.append(ToolMessage(content=result, name="dice_roll"))
            await CLMessage(content=result).send()

        elif mapped_action == "search" and WEB_SEARCH_ENABLED:
            query = last_human_message.content
            result = await web_search.ainvoke({"query": query})
            state.messages.append(ToolMessage(content=result, name="web_search"))
            await CLMessage(content=result).send()

        # Generate AI response
        ai_response = await generate_story_response(state.messages)
        state.messages.append(AIMessage(content=ai_response))
        await CLMessage(content=ai_response).send()

        # Generate storyboard if needed and image generation is enabled
        if IMAGE_GENERATION_ENABLED:
            storyboard = await process_storyboard(state)
            if storyboard:
                state.metadata["storyboard"] = storyboard
                await process_storyboard_images(storyboard, state.current_message_id)

        # Save state
        await save_chat_memory(state, store)

    except Exception as e:
        cl_logger.error(f"Error in chat workflow: {str(e)}", exc_info=True)
        state.increment_error_count()
        await CLMessage(content="⚠️ An error occurred while generating the response. Please try again later.").send()

    if writer:
        writer("Processing completed")

    return state
