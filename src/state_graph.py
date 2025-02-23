import asyncio
from typing import Dict, Any, List, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.store.base import BaseStore
from stores import VectorStore
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage
)
from langchain.prompts import PromptTemplate
import chainlit as cl
from chainlit import Message as CLMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from models import ChatState, MessageType
from tools_and_agents import (
    writer_agent,
    storyboard_editor_agent,
    dice_roll,
    web_search,
    decision_agent
)
from image_generation import (
    async_range,
    handle_image_generation,
    generate_image_generation_prompts
)

from config import (
    AI_WRITER_PROMPT,
    STORYBOARD_GENERATION_PROMPT,
    NUM_IMAGE_PROMPTS,
    REFUSAL_LIST,
    DECISION_PROMPT,
    SEARCH_ENABLED,
    DICE_SIDES,
)

@task
async def determine_action(state: ChatState) -> str:
    """Determine next action using LangGraph task."""
    try:
        last_message = state.messages[-1]
        if not isinstance(last_message, HumanMessage):
            return "writer"

        # Format messages for decision agent
        messages = [
            SystemMessage(content=DECISION_PROMPT),
            last_message
        ]
        
        # Use the decision agent directly
        response = await decision_agent.ainvoke(messages)
        
        action_map = {
            "roll": "roll",
            "search": "search" if SEARCH_ENABLED else "writer",
            "continue_story": "writer"
        }
        
        return action_map.get(response.action, "writer")
        
    except Exception as e:
        cl.logger.error(f"Error determining action: {e}")
        return "writer"

@task
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def handle_dice_roll(state: ChatState) -> Dict[str, Any]:
    """Handle dice roll request with retry logic."""
    result = await dice_roll()
    return {"messages": [ToolMessage(content=result)]}

@task
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def handle_search(state: ChatState) -> Dict[str, Any]:
    """Handle web search request with retry logic."""
    query = state.messages[-1].content
    result = await web_search(query)
    return {"messages": [ToolMessage(content=result)]}

@task
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_story_response(state: ChatState) -> Dict[str, Any]:
    """Generate main story response with retry logic."""
    try:
        response = await writer_agent.ainvoke(state.messages)
        return {"messages": [response]}
    except Exception as e:
        state.increment_error_count()
        if state.error_count >= 3:
            return {"messages": [SystemMessage(content="I apologize, but I'm having trouble generating a response. Please try again.")]}
        raise

@task
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_storyboard(state: ChatState, store: BaseStore) -> Optional[str]:
    """Generate storyboard for visualization with retry logic."""
    try:
        # Get relevant documents using the store
        docs = store.get((state.thread_id,), state.messages[-1].content)
        memories_str = "\n".join([doc.page_content for doc in docs]) if docs else ""
        
        # Format recent chat history properly
        recent_messages = state.get_recent_history()
        recent_chat_history = "\n".join([
            f"{msg.__class__.__name__.replace('Message', '').upper()}: {msg.content}" 
            for msg in recent_messages
        ])

        # Create the prompt with proper formatting
        prompt = PromptTemplate(
            template=STORYBOARD_GENERATION_PROMPT,
            input_variables=["memories", "recent_chat_history"]
        )
        
        formatted_prompt = prompt.format(
            memories=memories_str,
            recent_chat_history=recent_chat_history
        )
        
        # Create messages list with proper typing
        messages = [SystemMessage(content=formatted_prompt)]
        
        # Invoke the agent with proper async call
        try:
            storyboard_message = await storyboard_editor_agent.ainvoke(messages)
            
            if isinstance(storyboard_message, BaseMessage):
                content = storyboard_message.content
                if content:
                    # Process the response
                    think_end = "</think>"
                    if think_end in content:
                        content = content.split(think_end)[1].strip()
                    return content if content.strip() else None
            
            cl.logger.warning("Invalid storyboard message format")
            return None
            
        except Exception as e:
            cl.logger.error(f"Storyboard agent invocation failed: {e}")
            return None
            
    except Exception as e:
        cl.logger.error(f"Storyboard generation failed: {e}")
        return None

@task
async def handle_tools(state: ChatState, action: str) -> Dict[str, Any]:
    """Handle tool execution using LangGraph's ToolNode."""
    if action not in ["roll", "search"]:
        return {"messages": []}
        
    try:
        result = await tool_node.ainvoke(state)
        return result
    except Exception as e:
        cl.logger.error(f"Tool execution failed: {e}")
        return {"messages": []}

@entrypoint(checkpointer=MemorySaver(), store=VectorStore())
async def story_workflow(
    state: ChatState,
    *,
    store: BaseStore,
    previous: Any = None
) -> entrypoint.final[Dict[str, Any], ChatState]:
    """Main workflow using LangGraph constructs."""
    try:
        # Determine action using task
        action = await determine_action(state)
        cl.logger.info(f"Determined action: {action}")
        
        # Handle tools using ToolNode task
        tool_result = await handle_tools(state, action)
        if tool_result.get("messages"):
            state.messages.extend(tool_result["messages"])
            
        # Generate story response
        response = await generate_story_response(state)
        if response and "messages" in response:
            state.messages.extend(response["messages"])
        
        # Generate storyboard and images
        storyboard = None
        if state.messages:
            try:
                storyboard = await generate_storyboard(state, store)
                if storyboard:
                    image_prompts = await generate_image_generation_prompts(storyboard)
                    if image_prompts and state.metadata.get("current_message_id"):
                        asyncio.create_task(handle_image_generation(
                            image_prompts, 
                            state.metadata["current_message_id"]
                        ))
            except Exception as e:
                cl.logger.error(f"Storyboard/image generation failed: {e}")
        
        # Create new state
        new_state = ChatState(
            messages=state.messages,
            thread_id=state.thread_id,
            metadata={
                "last_action": action,
                "storyboard": storyboard,
                "current_message_id": state.metadata.get("current_message_id")
            }
        )
        
        return entrypoint.final(
            value={
                "messages": state.messages,
                "action": action,
                "storyboard": storyboard
            },
            save=new_state
        )
        
    except Exception as e:
        cl.logger.error(f"Workflow error: {e}")
        return entrypoint.final(
            value={
                "messages": [SystemMessage(content="I apologize, but I encountered an error. Please try again.")],
                "action": "error",
                "storyboard": None
            },
            save=state
        )

# Initialize the workflow
graph = story_workflow

# Export the necessary functions
__all__ = ['story_workflow', 'generate_storyboard', 'handle_image_generation']
