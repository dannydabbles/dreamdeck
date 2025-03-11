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
from chainlit import Message as CLMessage  # Import Chainlit's UI-facing message
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
            storyboard_result = await storyboard_editor_agent.generate_storyboard(
                last_human_message.content, state=state
            ).result()
            if storyboard_result:
                state.metadata["storyboard"] = storyboard_result

        # Convert LangGraph messages to Chainlit-compatible format for display
        cl_messages = [
            CLMessage(
                content=msg.content,
                author="ai" if isinstance(msg, AIMessage) else "user" if isinstance(msg, HumanMessage) else "tool",
                tool_call_id=msg.tool_call_id if isinstance(msg, ToolMessage) else None,
            )
            for msg in state.messages
        ]
        # Send to Chainlit UI
        for cl_msg in cl_messages:
            await cl_msg.send()

        # Save state
        await save_chat_memory(state, store)

    except Exception as e:
        cl_logger.error(f"Critical error in chat workflow: {str(e)}", exc_info=True)
        state.increment_error_count()
        state.messages.append(AIMessage(content="⚠️ A critical error occurred. Please try again later or restart the session."))

    return state


