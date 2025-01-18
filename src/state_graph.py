import asyncio
from typing import Annotated, List, Literal, TypedDict

from chainlit import Message as CLMessage
from image_generation import async_range, handle_image_generation, generate_image_generation_prompts
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import MessagesState

from langchain_core.messages.system import SystemMessage
from langchain_core.messages.modifier import RemoveMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.ai import AIMessage

from langchain.prompts import PromptTemplate

from tools_and_agents import writer_agent, storyboard_generation_agent

import chainlit as cl

from config import (
    AI_WRITER_PROMPT,
    STORYBOARD_GENERATION_PROMPT,
    NUM_IMAGE_PROMPTS,
    REFUSAL_LIST
)

# Define the state graph
builder = StateGraph(MessagesState)

# Define decision function to route messages
def start_router(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]

    return "writer"

# Define the model invocation functions
async def call_writer(state: MessagesState):
    # Extract necessary information from the state
    context = state["messages"]

    memories = cl.user_session.get("vector_memory", None).retriever.vectorstore.similarity_search(context[-1].content, 5)

    # Format the system prompt using the PromptTemplate
    engrams=list(set([memory.page_content for memory in memories]))
    if engrams:
        memories="\n".join(engrams)
    else:
        memories="No additional inspiration provided"
    recent_chat_history="\n".join([f"{message.type.upper()}: {message.content}" for message in context])
    writer_prompt = PromptTemplate.from_template(AI_WRITER_PROMPT)
    system_content = writer_prompt.format(
        memories=memories,
        recent_chat_history=recent_chat_history,
    )

    # Create the message list with the system message and the latest user message
    messages = [SystemMessage(content=system_content)]

    response = None
    while not response:
        try:
            # Invoke the writer agent with the formatted messages
            response = await writer_agent.ainvoke(messages)

            # Get last message from the response
            new_message = response.content

            # Check against refusal list or copying the question
            if any([new_message.strip().startswith(refusal) for refusal in REFUSAL_LIST]) or new_message == context[-1].content:
                response = None
                raise ValueError("Writer model refused to generate message: {new_message}")

        except Exception as e:
            cl.logger.error(f"Writer model invocation failed: {e}")
            response = None

    return {"messages": [response]}

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
        memories="\n".join(engrams)
    else:
        memories="No additional inspiration provided."
    recent_chat_history="\n".join([f"{message.type.upper()}: {message.content}" for message in context])

    image_prompt = PromptTemplate.from_template(STORYBOARD_GENERATION_PROMPT)
    system_content = image_prompt.format(
        memories=memories,
        recent_chat_history=recent_chat_history
    )
    # Create the message list with the system message and the latest user message
    messages = [SystemMessage(content=system_content)]

    storyboard_message = await storyboard_generation_agent.ainvoke(messages)

    return storyboard_message.content

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

# Define edges based on the decision function
builder.add_conditional_edges(
    START,
    start_router,
)
builder.add_edge("writer", END)

graph = builder.compile()
