import asyncio
import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from langgraph.types import StreamWriter
from langchain_core.runnables import RunnableConfig
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, ToolExecutor
from langgraph.store.base import BaseStore
from stores import VectorStore
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    BaseMessage
)

# Initialize logging
cl_logger = logging.getLogger("chainlit")
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
    decision_agent,
    log_decision_agent_response
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
    LLM_TIMEOUT,
)

@task
async def determine_action(state: ChatState) -> str:
    """Determine next action using LangGraph task."""
    try:
        last_message = state.messages[-1]
        if not isinstance(last_message, HumanMessage):
            cl.logger.info("Last message not a HumanMessage, defaulting to writer")
            return "writer"

        # Format messages for decision agent
        messages = [
            SystemMessage(content=DECISION_PROMPT),
            last_message
        ]
        
        cl.logger.debug(f"Sending messages to decision agent: {messages}")
        
        try:
            response = await decision_agent.ainvoke(messages)
            log_decision_agent_response(response)  # Log detailed response info
            
            # Extract the function call result
            if (
                hasattr(response, 'additional_kwargs') 
                and 'function_call' in response.additional_kwargs
            ):
                function_call = response.additional_kwargs['function_call']
                cl.logger.debug(f"Function call details: {function_call}")
                
                if function_call.get('name') == 'decide_action':
                    import json
                    try:
                        args = json.loads(function_call['arguments'])
                        action = args.get('action')
                        cl.logger.info(f"Decision agent returned action: {action}")
                    except json.JSONDecodeError as e:
                        cl.logger.error(f"Failed to parse function arguments: {e}")
                        cl.logger.debug(f"Raw arguments: {function_call['arguments']}")
                        action = 'continue_story'
                else:
                    cl.logger.warning(f"Unexpected function call: {function_call.get('name')}")
                    action = 'continue_story'
            else:
                cl.logger.warning("No function call in response")
                # Try to extract from content as fallback
                content = response.content.lower().strip() if hasattr(response, 'content') else ''
                cl.logger.debug(f"Attempting to extract action from content: {content}")
                if "roll" in content:
                    action = "roll"
                elif "search" in content:
                    action = "search"
                else:
                    action = "continue_story"
            
            action_map = {
                "roll": "roll",
                "search": "search" if SEARCH_ENABLED else "writer",
                "continue_story": "writer"
            }
            
            mapped_action = action_map.get(action, "writer")
            cl.logger.info(f"Mapped action to: {mapped_action}")
            return mapped_action
            
        except Exception as e:
            cl.logger.error(f"Error in decision agent: {e}", exc_info=True)  # Include stack trace
            return "writer"
            
    except Exception as e:
        cl.logger.error(f"Error determining action: {e}", exc_info=True)  # Include stack trace
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
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((TypeError, ValueError))
)
async def generate_story_response(state: ChatState) -> Dict[str, Any]:
    """Generate main story response with retry logic."""
    try:
        messages = []
        # Convert state messages to dict format that OpenAI can handle
        input_messages = [
            {
                "role": "system" if isinstance(msg, SystemMessage) else 
                        "assistant" if isinstance(msg, AIMessage) else
                        "function" if isinstance(msg, ToolMessage) else
                        "user",
                "content": msg.content
            }
            for msg in state.messages
        ]
        
        # Create config for streaming
        config = RunnableConfig(
            callbacks=None,
            tags=["story"],
            metadata={"type": "story_generation"}
        )
        
        async for chunk in writer_agent.astream(
            input=input_messages,
            config=config
        ):
            if isinstance(chunk, BaseMessage):
                messages.append(chunk)
            elif isinstance(chunk, dict) and chunk.get("content"):
                messages.append(AIMessage(content=chunk["content"]))
        
        if not messages:
            raise ValueError("No messages generated")
            
        return {"messages": messages}
        
    except Exception as e:
        cl_logger.error(f"Story generation error: {str(e)}", exc_info=True)
        state.increment_error_count()
        if state.error_count >= 3:
            return {
                "messages": [
                    SystemMessage(content="I apologize, but I'm having trouble generating a response. Please try again.")
                ]
            }
        raise

@task
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def generate_storyboard(state: ChatState, store: BaseStore) -> Optional[str]:
    """Generate storyboard for visualization with retry logic."""
    try:
        cl.logger.debug("Starting storyboard generation")
        
        # Get relevant documents using the store
        docs = store.get((state.thread_id,), state.messages[-1].content)
        memories_str = "\n".join([d.page_content for d in docs]) if docs else ""
        
        # Format recent chat history properly
        recent_messages = state.get_recent_history()
        recent_chat_history = "\n".join([
            f"{msg.__class__.__name__.replace('Message', '').upper()}: {msg.content}" 
            for msg in recent_messages
        ])

        # Create messages list with proper typing
        messages = [
            SystemMessage(content=STORYBOARD_GENERATION_PROMPT.format(
                memories=memories_str,
                recent_chat_history=recent_chat_history
            ))
        ]
        
        try:
            # Use the storyboard editor agent with proper async call
            response = await storyboard_editor_agent.ainvoke(
                messages,
                timeout=LLM_TIMEOUT * 2  # Give more time for storyboard generation
            )
            cl.logger.debug(f"Raw storyboard response type: {type(response)}")
            cl.logger.debug(f"Raw storyboard response: {response}")
            
            if isinstance(response, BaseMessage):
                content = response.content
                if content:
                    # Process the response
                    think_end = "</think>"
                    if think_end in content:
                        content = content.split(think_end)[1].strip()
                    
                    if content.strip():
                        return content
                        
            cl.logger.warning("Invalid or empty storyboard response")
            return None
                
            # Process the response
            think_end = "</think>"
            if think_end in content:
                content = content.split(think_end)[1].strip()
            
            if not content.strip():
                cl.logger.warning("Empty content after processing")
                return None
                
            return content
            
        except Exception as e:
            cl.logger.error(f"Storyboard agent invocation failed: {str(e)}", exc_info=True)
            return None
            
    except Exception as e:
        cl.logger.error(f"Storyboard outer error: {str(e)}", exc_info=True)
        return None

@task
async def handle_tools(state: ChatState, action: str) -> Dict[str, Any]:
    """Handle tool execution using LangGraph's ToolNode."""
    if action not in ["roll", "search"]:
        return {"messages": []}
        
    try:
        if action == "roll":
            # Pass None as default input for dice roll
            result = await dice_roll.ainvoke({"n": None})  # Use ainvoke instead of direct call
            return {"messages": [ToolMessage(content=str(result))]}
        elif action == "search":
            query = state.messages[-1].content
            result = await web_search.ainvoke({"query": query})  # Use ainvoke for web search too
            return {"messages": [ToolMessage(content=str(result))]}
            
    except Exception as e:
        cl.logger.error(f"Tool execution failed: {e}", exc_info=True)
        return {"messages": []}

@entrypoint(checkpointer=MemorySaver(), store=VectorStore())
async def story_workflow(
    state: ChatState,
    *,
    store: BaseStore,
    previous: Any = None,
    writer: StreamWriter = None
) -> entrypoint.final[Dict[str, Any], ChatState]:
    """Main workflow using LangGraph constructs."""
    try:
        if writer:
            writer("Processing started")
            
        # Initialize messages from previous state if available
        if previous:
            state.messages.extend(previous.messages)

        # Determine action
        action = await determine_action(state)
        cl.logger.info(f"Determined action: {action}")

        # Handle dice roll
        roll_result = None
        if action == "roll":
            roll_result = await handle_dice_roll(state)
            if roll_result.get("messages"):
                # Add roll result to chat history
                roll_message = roll_result["messages"][0]
                state.messages.append(roll_message)
                
                # Display roll in chat
                await cl.Message(content=roll_message.content).send()

        # Handle search silently if needed
        search_result = None
        if action == "search":
            search_result = await handle_search(state)
            # Don't add search result to messages/chat history
            # It will be used internally by the GM

        # Generate GM response
        msg = cl.Message(content="")
        
        try:
            # If we have tool results, add them to state
            if roll_result or search_result:
                if roll_result:
                    state.add_tool_result(roll_result["messages"][0].content)
                if search_result:
                    state.add_tool_result(search_result["messages"][0].content)

            # Get response
            response = await generate_story_response(state)
            
            # Stream the response content
            if response and response.get("messages"):
                for message in response["messages"]:
                    if isinstance(message, (AIMessage, SystemMessage)):
                        await msg.stream_token(message.content)
                    
            await msg.send()
            
            # Update state with GM response
            if msg.content:
                state.messages.append(AIMessage(content=msg.content))
                state.metadata["current_message_id"] = msg.id
                
        except Exception as e:
            cl_logger.error(f"Story generation failed: {str(e)}", exc_info=True)
            await cl.Message(content="I apologize, but I encountered an error. Please try again.").send()
            return entrypoint.final(
                value={
                    "messages": [SystemMessage(content="I apologize, but I encountered an error. Please try again.")],
                    "action": "error"
                },
                save=state
            )
        
        # Clear tool results after they've been used
        state.clear_tool_results()
        
        # Update state with GM response
        if msg.content:
            state.messages.append(AIMessage(content=msg.content))
            state.metadata["current_message_id"] = msg.id

        # Generate and display storyboard images asynchronously
        if state.messages:
            try:
                storyboard = await generate_storyboard(state, store)
                if storyboard:
                    image_prompts = await generate_image_generation_prompts(storyboard)
                    if image_prompts and state.metadata.get("current_message_id"):
                        # Don't add storyboard text to history
                        asyncio.create_task(handle_image_generation(
                            image_prompts,
                            state.metadata["current_message_id"]
                        ))
            except Exception as e:
                cl.logger.error(f"Storyboard/image generation failed: {e}")

        if writer:
            writer("Processing completed")
            
        return entrypoint.final(
            value={
                "messages": state.messages[-1:],
                "action": action
            },
            save=state
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
