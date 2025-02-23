import asyncio
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.graph.message import MessagesState

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

# Define the state graph
builder = StateGraph(MessagesState)

async def determine_next_action(state: ChatState) -> str:
    """Determine the next action based on the latest message using the decision agent."""
    try:
        last_message = state.messages[-1]
        if last_message.type != MessageType.HUMAN:
            return "writer"

        # Get decision from agent
        response = await decision_agent.ainvoke({
            "input": last_message.content,
            "chat_history": state.get_recent_history()
        })
        
        action = response.get("output", "writer").strip().lower()
        
        # Map actions to handlers
        action_map = {
            "roll": "roll",
            "search": "search" if SEARCH_ENABLED else "writer",
            "continue_story": "writer"
        }
        
        return action_map.get(action, "writer")
        
    except Exception as e:
        cl.logger.error(f"Error determining next action: {e}")
        return "writer"

async def start_router(state: MessagesState):
    action = await determine_next_action(state)
    return action  # Ensure the action key is set in the metadata

# Define the model invocation functions
async def call_writer(state: MessagesState):
    # Extract necessary information from the state
    context = state["messages"]

    # Filter out ToolMessages since the last user message
    tool_results = [msg for msg in context if isinstance(msg, ToolMessage)]
    story_messages = [msg for msg in context if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)]

    # Get the most recent chat history
    memories = cl.user_session.get("vector_memory", None).retriever.vectorstore.similarity_search(story_messages[-1].content, 5)

    # Get the most recent chat history
    recent_chat_history = "\n".join([f"{message.type.upper()}: {message.content}" for message in story_messages])
    tool_results_str = "\n".join(tool_results) if tool_results else "No tool results"

    # Add memories relating to the last message
    engrams = []
    for memory in memories:
        # Check if the memory is already in the engrams list or in the chat history
        if memory.page_content.strip() not in engrams and memory.page_content.strip() not in recent_chat_history:
            engrams.append(memory.page_content)
    if engrams:
        memories = "\n".join(engrams)
    else:
        memories = "No additional inspiration provided"

    # Format the system prompt using the PromptTemplate
    writer_prompt = PromptTemplate.from_template(AI_WRITER_PROMPT)
    system_content = writer_prompt.format(
        memories=memories,
        recent_chat_history=recent_chat_history,
        tool_results=tool_results_str
    )

    # Create the message list with the system message and the latest user message
    messages = [SystemMessage(content=system_content)]

    # Invoke the writer agent
    response = None
    while not response:
        try:
            # Invoke the writer agent with the formatted messages
            response = await writer_agent.ainvoke(messages)

            # Get last message from the response
            new_message = response.content

            # Check against refusal list or copying the question
            if any([new_message.strip().startswith(refusal) for refusal in REFUSAL_LIST]) or new_message == story_messages[-1].content:
                response = None
                raise ValueError("Writer model refused to generate message: {new_message}")

        except Exception as e:
            cl.logger.error(f"Writer model invocation failed: {e}")
            response = None

    # Ensure the metadata includes the 'action' key
    return {"messages": [response]}

async def call_dice_roll(state: MessagesState):
    last_message = state["messages"][-1]
    try:
        # Extract the number of sides from the message content
        sides = int(last_message.content.split(" ")[1])
    except (ValueError, IndexError):
        sides = DICE_SIDES
    result = dice_roll(sides)
    tool_output = {
        "stdout": result,
        "stderr": None,
        "artifacts": None,
    }
    return {"messages": [ToolMessage(content=tool_output["stdout"], artifact=tool_output, tool_call_id="roll_dice")]}

async def call_web_search(state: MessagesState):
    last_message = state["messages"][-1]
    query = last_message.content.replace("search ", "").strip()
    result = web_search(query)
    tool_output = {
        "stdout": result,
        "stderr": None,
        "artifacts": None,
    }
    return {"messages": [ToolMessage(content=tool_output["stdout"], artifact=tool_output, tool_call_id="web_search")]}

async def call_storyboard_generation(history: List[BaseMessage]):
    context = history
    # Add memories relating to the last message
    memories = cl.user_session.get("vector_memory", None).retriever.vectorstore.similarity_search(context[-1].content, 3)
    if len(context) > 1:
        # Add memories relating to the second to last message
        memories = cl.user_session.get("vector_memory", None).retriever.vectorstore.similarity_search(context[-2].content, 3)
    # Format the system prompt using the PromptTemplate
    engrams = list(set([memory.page_content for memory in memories]))
    if engrams:
        memories = "\n".join(engrams)
    else:
        memories = "No additional inspiration provided."
    recent_chat_history = "\n".join([f"{message.type.upper()}: {message.content}" for message in context])

    image_prompt = PromptTemplate.from_template(STORYBOARD_GENERATION_PROMPT)
    system_content = image_prompt.format(
        memories=memories,
        recent_chat_history=recent_chat_history
    )
    # Create the message list with the system message and the latest user message
    messages = [SystemMessage(content=system_content)]

    storyboard_message = await storyboard_editor_agent.ainvoke(messages)

    think_end = "</think>"
    storyboard = storyboard_message.content
    if think_end in storyboard:
        storyboard = storyboard.split(think_end)[1].strip()

    return storyboard

async def call_image_generation(state: MessagesState, ai_message_id: str):
    storyboard = await call_storyboard_generation(state)
    image_gen_prompts = []
    try:
        image_gen_prompts = await generate_image_generation_prompts(storyboard)
    except ValueError:
        cl.logger.warning("LLM refused to generate image prompt. Skipping this image generation.")
        image_gen_prompts = []

    if image_gen_prompts:
        # Schedule image generation
        await handle_image_generation(image_gen_prompts, ai_message_id)
    else:
        cl.logger.warning("Image generation failed!")
        await CLMessage(content="⚠️ Image generation failed!", parent_id=ai_message_id).send()

# Add nodes to the graph
builder.add_node("writer", call_writer)
builder.add_node("roll", call_dice_roll)
builder.add_node("search", call_web_search)

# Define edges based on the decision function
builder.add_conditional_edges(
    START,
    start_router,
)
builder.add_edge("writer", END)
builder.add_edge("roll", "writer")
builder.add_edge("search", "writer")

graph = builder.compile()
