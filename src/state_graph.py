import asyncio
import logging
import json
from typing import List, Optional
from langgraph.prebuilt import create_react_agent
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage
)
from chainlit import Message as CLMessage  # Import CLMessage from Chainlit
from .state import ChatState
from .agents.decision_agent import decide_action  # Import decide_action
from .agents.dice_agent import dice_roll  # Import dice_roll
from .agents.web_search_agent import web_search  # Import web_search
from .agents.writer_agent import generate_story  # Import generate_story
from .agents.storyboard_editor_agent import generate_storyboard  # Import generate_storyboard
from .image_generation import process_storyboard_images
from .config import (
    IMAGE_GENERATION_ENABLED,
    DECISION_PROMPT,  # Import DECISION_PROMPT
)
from .models import ChatState
from .memory_management import save_chat_memory
from .stores import BaseStore  # Import BaseStore

# Initialize logging
cl_logger = logging.getLogger("chainlit")


@task
async def process_storyboard(state: ChatState) -> Optional[str]:
    """Generate storyboard prompts based on the GM's last response.

    Args:
        state (ChatState): The current chat state.

    Returns:
        Optional[str]: The generated storyboard or None if generation fails.
    """
    try:
        # Get the last GM response
        last_gm_message = next(
            (msg for msg in reversed(state.messages) if isinstance(msg, AIMessage)),
            None,
        )
        if not last_gm_message:
            cl_logger.warning("No GM message found to generate storyboard from")
            return None

        # Format the prompt with proper context
        formatted_prompt = DECISION_PROMPT.format(
            recent_chat_history=state.get_recent_history_str(),
            memories=state.get_memories_str(),
        )

        # Create messages list for the storyboard generation
        messages = [
            SystemMessage(content=formatted_prompt),
            HumanMessage(content=last_gm_message.content),
        ]

        # Get storyboard prompts from the agent
        response = await generate_storyboard(last_gm_message.content).result()

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
) -> ChatState:
    """Main chat workflow handling messages and state.

    Args:
        messages (List[BaseMessage]): List of incoming messages.
        store (BaseStore): The store for chat state.
        previous (Optional[ChatState], optional): Previous chat state. Defaults to None.

    Returns:
        ChatState: The updated chat state.
    """
    state = previous or ChatState()
    state.messages.extend(messages)

    try:
        # Determine action
        last_human_message = next(
            (msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
            None,
        )
        if not last_human_message:
            cl_logger.info("No human message found, defaulting to continue_story")
            action = "continue_story"
        else:
            formatted_prompt = DECISION_PROMPT.format(
                user_input=last_human_message.content
            )
            decision_response = await decide_action(formatted_prompt).result()
            action = decision_response.get("name", "continue_story")

        action_map = {
            "roll": dice_roll,
            "search": web_search,
            "continue_story": generate_story,
        }
        mapped_task = action_map.get(action, generate_story)

        if mapped_task == dice_roll:
            result = await dice_roll(last_human_message.content).result()
            tool_message = ToolMessage(content=result.content, name="dice_roll")
            state.add_tool_message(tool_message)
            state.messages.append(tool_message)

        elif mapped_task == web_search:
            result = await web_search(last_human_message.content).result()
            tool_message = ToolMessage(content=result.content, name="web_search")
            state.add_tool_message(tool_message)
            state.messages.append(tool_message)

        else:
            ai_response = await generate_story(last_human_message.content).result()
            state.messages.append(AIMessage(content=ai_response))

        # Generate storyboard if needed and image generation is enabled
        if IMAGE_GENERATION_ENABLED:
            storyboard = await process_storyboard(state).result()
            if storyboard:
                state.metadata["storyboard"] = storyboard
                await process_storyboard_images(storyboard, state.current_message_id).result()

        # Save state
        await save_chat_memory(state, store)

    except Exception as e:
        cl_logger.error(f"Critical error in chat workflow: {str(e)}", exc_info=True)
        state.increment_error_count()
        state.messages.append(AIMessage(content="⚠️ A critical error occurred. Please try again later or restart the session."))

    return state


async def generate_story_response(messages: List[BaseMessage]) -> str:
    """Generate a story response from the given messages."""
    try:
        # TODO: Implement actual story generation logic
        # For now, just return a placeholder response
        return "This is a placeholder story response."
    except Exception as e:
        cl_logger.error(f"Error generating story response: {str(e)}", exc_info=True)
        return "Error generating response."
