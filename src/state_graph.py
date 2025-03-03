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
    ToolMessage,
)
from langgraph.message import CLMessage
from .state import ChatState
from .agents.decision_agent import decision_agent
from .agents.writer_agent import writer_agent
from .agents.storyboard_editor_agent import storyboard_editor_agent
from .agents.dice_agent import dice_roll_agent
from .agents.web_search_agent import web_search_agent
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
    writer: asyncio.StreamWriter = None,
) -> ChatState:
    """Main chat workflow handling messages and state.

    Args:
        messages (List[BaseMessage]): List of incoming messages.
        store (BaseStore): The store for chat state.
        previous (Optional[ChatState], optional): Previous chat state. Defaults to None.
        writer (asyncio.StreamWriter, optional): Stream writer for logging. Defaults to None.

    Returns:
        ChatState: The updated chat state.
    """
    state = previous or ChatState()
    state.messages.extend(messages)

    try:
        if writer:
            writer("Processing started")

        # Determine action
        last_human_message = next(
            (msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)),
            None,
        )
        if not last_human_message:
            cl_logger.info("No human message found, defaulting to writer")
            action = "continue_story"
        else:
            formatted_prompt = DECISION_PROMPT.format(
                user_input=last_human_message.content
            )
            messages = [SystemMessage(content=formatted_prompt)]
            response = await decision_agent.ainvoke(messages)

            if (
                hasattr(response, "additional_kwargs")
                and "function_call" in response.additional_kwargs
            ):
                function_call = response.additional_kwargs["function_call"]
                try:
                    args = json.loads(function_call["arguments"])
                    action = args.get("action")
                except json.JSONDecodeError as e:
                    cl_logger.error(f"Failed to parse function arguments: {e}")
                    action = "continue_story"
            else:
                action = "continue_story"

        action_map = {
            "roll": dice_roll_agent,
            "search": web_search_agent,
            "continue_story": writer_agent,
        }
        mapped_agent = action_map.get(action, writer_agent)

        if mapped_agent == dice_roll_agent:
            result = await dice_roll_agent.ainvoke([HumanMessage(content=last_human_message.content)])
            state.messages.append(result)
            await CLMessage(content=result.content).send()

        elif mapped_agent == web_search_agent:
            result = await web_search_agent.ainvoke([HumanMessage(content=last_human_message.content)])
            state.messages.append(result)
            await CLMessage(content=result.content).send()

        else:
            ai_response = await mapped_agent.ainvoke(state.messages)
            state.messages.append(AIMessage(content=ai_response.content))
            await CLMessage(content=ai_response.content).send()

        # Generate storyboard if needed and image generation is enabled
        if IMAGE_GENERATION_ENABLED:
            storyboard = await process_storyboard(state)
            if storyboard:
                state.metadata["storyboard"] = storyboard
                await process_storyboard_images(storyboard, state.current_message_id)

        # Save state
        await save_chat_memory(state, store)

    except Exception as e:
        cl_logger.error(f"Critical error in chat workflow: {str(e)}", exc_info=True)
        state.increment_error_count()
        await CLMessage(
            content="⚠️ A critical error occurred. Please try again later or restart the session."
        ).send()
        raise

    if writer:
        writer("Processing completed")

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
