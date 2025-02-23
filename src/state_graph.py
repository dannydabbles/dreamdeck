import asyncio
import logging
import random
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
    process_storyboard_images,
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
        # Get the last human message
        last_human_message = None
        for msg in reversed(state.messages):
            if isinstance(msg, HumanMessage):
                last_human_message = msg
                break

        if not last_human_message:
            cl.logger.info("No human message found, defaulting to writer")
            return "writer"

        # Format the decision prompt with the user's input
        formatted_prompt = DECISION_PROMPT.format(
            user_input=last_human_message.content
        )
        
        cl.logger.debug(f"Formatted decision prompt: {formatted_prompt}")
        
        # Create messages list for decision agent
        messages = [
            SystemMessage(content=formatted_prompt)
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
                        
                        # Check for dice roll keywords in the message
                        if (
                            "roll" in last_human_message.content.lower() or 
                            "d20" in last_human_message.content.lower() or
                            "dice" in last_human_message.content.lower()
                        ):
                            cl.logger.info("Dice roll detected in message, forcing roll action")
                            action = "roll"
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
            cl.logger.error(f"Error in decision agent: {e}", exc_info=True)
            return "writer"
            
    except Exception as e:
        cl.logger.error(f"Error determining action: {e}", exc_info=True)
        return "writer"

@task
async def handle_dice_roll(state: ChatState) -> Dict[str, Any]:
    """Handle dice roll request."""
    try:
        # Extract number from message like "Roll a d20"
        message = state.messages[-1].content.lower()
        n = None
        if 'd' in message:
            try:
                n = int(message.split('d')[1].split()[0])
            except (IndexError, ValueError):
                n = 20  # Default to d20
                
        # Call dice_roll once
        result = await dice_roll.ainvoke({"n": n})
        
        # Create the roll message with proper formatting
        roll_message = ToolMessage(
            content=result,
            additional_kwargs={
                "name": "dice_roll",
                "type": "tool",
                "tool_call_id": f"dice_roll_{random.randint(0, 1000000)}"  # Add unique tool_call_id
            }
        )
        
        # Log the result
        cl_logger.info(f"Dice roll completed: {result}")
        
        # Add roll result to state's tool results for GM context
        state.add_tool_result(result)
        
        return {"messages": [roll_message]}
        
    except Exception as e:
        cl_logger.error(f"Handle dice roll failed: {e}")
        error_msg = "🎲 Error handling dice roll. Using default d20."
        return {
            "messages": [
                ToolMessage(
                    content=error_msg,
                    additional_kwargs={
                        "name": "dice_roll",
                        "type": "tool",
                        "tool_call_id": f"dice_roll_error_{random.randint(0, 1000000)}"  # Add unique tool_call_id for error case
                    }
                )
            ]
        }

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
async def generate_storyboard(state: ChatState) -> Optional[str]:
    """Generate storyboard prompts based on the GM's last response."""
    try:
        # Get the last GM response
        last_gm_message = None
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                last_gm_message = msg
                break
                
        if not last_gm_message:
            cl.logger.warning("No GM message found to generate storyboard from")
            return None

        # Format the prompt with proper context
        formatted_prompt = STORYBOARD_GENERATION_PROMPT.format(
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
            cl.logger.warning("Empty storyboard response")
            return None
            
        # Clean up the response - remove any thinking tags, etc.
        content = response.content
        if "</think>" in content:
            content = content.split("</think>")[1].strip()
            
        return content.strip()
            
    except Exception as e:
        cl.logger.error(f"Storyboard generation failed: {str(e)}", exc_info=True)
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

        # Initialize result variables
        roll_result = None
        search_result = None

        # Determine action
        action = await determine_action(state)
        cl.logger.info(f"Determined action: {action}")

        # Handle dice roll
        if action == "roll":
            try:
                roll_result = await handle_dice_roll(state)
                if roll_result and roll_result.get("messages"):
                    roll_message = roll_result["messages"][0]
                    
                    # Add to state messages
                    state.messages.append(roll_message)
                    
                    # Send to chat
                    await cl.Message(content=roll_message.content).send()
                    
                    # Return early after dice roll
                    return entrypoint.final(
                        value={
                            "messages": [roll_message],
                            "action": action
                        },
                        save=state
                    )
            except Exception as e:
                cl_logger.error(f"Dice roll handling failed: {e}")
                error_msg = "🎲 Error handling dice roll."
                await cl.Message(content=error_msg).send()
                return entrypoint.final(
                    value={
                        "messages": [SystemMessage(content=error_msg)],
                        "action": "error"
                    },
                    save=state
                )

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

            # Generate storyboard immediately after GM response
            storyboard = await generate_storyboard(state)
            
            # Process storyboard images asynchronously if we have content
            if storyboard and state.metadata.get("current_message_id"):
                try:
                    await process_storyboard_images(
                        storyboard,
                        state.metadata["current_message_id"]
                    )
                except Exception as e:
                    cl_logger.error(f"Failed to process storyboard images: {e}")
                
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
__all__ = ['story_workflow', 'generate_storyboard', 'process_storyboard_images']
