import asyncio
from typing import Dict, Any, List, Optional
from langgraph.func import entrypoint, task
from langgraph.checkpoint import MemorySaver
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
    """Determine next action based on user input using decision agent."""
    try:
        last_message = state.messages[-1]
        if last_message.type != MessageType.HUMAN:
            return "writer"

        response = await decision_agent.ainvoke({
            "input": last_message.content,
            "chat_history": state.get_recent_history()
        })
        
        action = response.get("output", "writer").strip().lower()
        action_map = {
            "roll": "roll",
            "search": "search" if SEARCH_ENABLED else "writer",
            "continue_story": "writer"
        }
        
        state.current_tool = action_map.get(action, "writer")
        state.clear_tool_results()
        
        return state.current_tool
        
    except Exception as e:
        cl.logger.error(f"Error determining action: {e}")
        state.increment_error_count()
        if state.error_count >= 3:
            return "error"
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
async def generate_storyboard(state: ChatState) -> Optional[str]:
    """Generate storyboard for visualization with retry logic."""
    try:
        context = state.messages
        memories = cl.user_session.get("vector_memory", None).retriever.vectorstore.similarity_search(context[-1].content, 3)
        
        engrams = list(set([memory.page_content for memory in memories]))
        memories_str = "\n".join(engrams) if engrams else "No additional inspiration provided."
        recent_chat_history = "\n".join([f"{message.type.upper()}: {message.content}" for message in context])

        image_prompt = PromptTemplate.from_template(STORYBOARD_GENERATION_PROMPT)
        system_content = image_prompt.format(
            memories=memories_str,
            recent_chat_history=recent_chat_history
        )
        
        messages = [SystemMessage(content=system_content)]
        storyboard_message = await storyboard_editor_agent.ainvoke(messages)
        
        think_end = "</think>"
        storyboard = storyboard_message.content
        if think_end in storyboard:
            storyboard = storyboard.split(think_end)[1].strip()
            
        return storyboard
    except Exception as e:
        cl.logger.error(f"Storyboard generation failed: {e}")
        return None

@entrypoint(checkpointer=MemorySaver(), store=VectorStore())
async def story_workflow(
    state: ChatState,
    *,
    store: BaseStore,
    previous: Any = None
) -> entrypoint.final[Dict[str, Any], ChatState]:
    """Main workflow with proper state management."""
    try:
        # Determine next action
        action = await determine_action(state)
        
        # Handle tools using ToolNode
        if action in ["roll", "search"]:
            tool_executor = ToolExecutor(tools=[dice_roll, web_search])
            tool_node = ToolNode(tools=tool_executor)
            tool_result = await tool_node.ainvoke(state)
            state.messages.extend(tool_result["messages"])
            
        # Generate story response
        response = await generate_story_response(state)
        state.messages.extend(response["messages"])
        
        # Generate storyboard and images
        storyboard = await generate_storyboard(state, store)
        
        # Save state for next interaction
        new_state = ChatState(
            messages=state.messages,
            vector_store_id=state.vector_store_id,
            metadata={"last_action": action, "storyboard": storyboard}
        )
        
        return entrypoint.final(
            value={"messages": state.messages, "action": action, "storyboard": storyboard},
            save=new_state
        )
    except Exception as e:
        cl.logger.error(f"Workflow error: {e}")
        state.increment_error_count()
        return {
            "messages": [SystemMessage(content="I apologize, but I encountered an error. Please try again.")],
            "action": "error",
            "storyboard": None
        }

# Initialize the workflow
graph = story_workflow
