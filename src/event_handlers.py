import os
import asyncio
import random
import base64
import httpx
from typing import List, Optional
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from chainlit import on_chat_start, on_chat_resume, on_message, user_session as cl_user_session, Message as CLMessage, element as cl_element, User, context
from chainlit.element import Image as CLImage, Select
from chainlit.types import ThreadDict, RunnableConfig
from chainlit import LangchainCallbackHandler
from langgraph.func import task
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage  # Import missing message types

from .config import (
    NEGATIVE_PROMPT,
    STEPS,
    SAMPLER_NAME,
    SCHEDULER,
    CFG_SCALE,
    DISTILLED_CFG_SCALE,
    WIDTH,
    HEIGHT,
    HR_UPSCALER,
    DENOISING_STRENGTH,
    HR_SECOND_PASS_STEPS,
    IMAGE_GENERATION_TIMEOUT,
    REFUSAL_LIST,
    KNOWLEDGE_DIRECTORY,
    STORYBOARD_GENERATION_PROMPT_PREFIX,
    STORYBOARD_GENERATION_PROMPT_POSTFIX,
    AI_WRITER_PROMPT,
    CHAINLIT_STARTERS,
    STABLE_DIFFUSION_API_URL,
    MAX_RETRIES,
    RETRY_DELAY,
    ERROR_LOG_LEVEL,
    IMAGE_GENERATION_RATE_LIMIT,
    API_CALLS_RATE_LIMIT,
    MONITORING_ENABLED,
    MONITORING_ENDPOINT,
    MONITORING_SAMPLE_RATE,
    CACHING_ENABLED,
    CACHING_TTL,
    CACHING_MAX_SIZE,
    IMAGE_GENERATION_ENABLED,
    WEB_SEARCH_ENABLED,
    DICE_ROLLING_ENABLED
)
from .state import ChatState
from .state_graph import chat_workflow as graph
from .tools_and_agents import handle_dice_roll  # Import handle_dice_roll
from .initialization import DatabasePool  # Import DatabasePool

# Define an asynchronous range generator
async def async_range(end):
    for i in range(0, end):
        # Sleep for a short duration to simulate asynchronous operation
        await asyncio.sleep(.1)
        yield i

async def generate_image_async(image_generation_prompt: str, seed: int) -> Optional[bytes]:
    """
    Generates an image asynchronously based on the image generation prompt using the Stable Diffusion API.

    Args:
        image_generation_prompt (str): The image generation prompt.
        seed (int): The seed for the image generation.

    Returns:
        Optional[bytes]: The image bytes, or None if generation fails.
    """
    if not IMAGE_GENERATION_ENABLED:
        cl_element.logger.warning("Image generation is disabled in the configuration.")
        return None

    # Flux payload
    payload = {
        "prompt": image_generation_prompt,
        "negative_prompt": NEGATIVE_PROMPT,
        "steps": STEPS,
        "sampler_name": SAMPLER_NAME,
        "scheduler": SCHEDULER,
        "cfg_scale": CFG_SCALE,
        "distilled_cfg_scale": DISTILLED_CFG_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "hr_upscaler": HR_UPSCALER,
        "denoising_strength": DENOISING_STRENGTH,
        "hr_second_pass_steps": HR_SECOND_PASS_STEPS,
        "seed": seed,
    }

    try:
        async with httpx.AsyncClient(timeout=IMAGE_GENERATION_TIMEOUT) as client:
            response = await client.post(f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            image_data = response.json()["images"][0]
            image_bytes = base64.b64decode(image_data)
            return image_bytes
    except httpx.RequestError as e:
        cl_element.logger.error(f"Image generation failed: {e}", exc_info=True)
        raise
    except (KeyError, IndexError, ValueError) as e:
        cl_element.logger.error(f"Error processing image data: {e}", exc_info=True)
        return None

@retry(
    wait=wait_exponential(multiplier=1, min=RETRY_DELAY, max=RETRY_DELAY * MAX_RETRIES),
    stop=stop_after_attempt(MAX_RETRIES),
    retry=retry_if_exception_type(httpx.RequestError)
)
async def generate_image_generation_prompts(
    storyboard: str
) -> List[str]:
    """
    Generates a list of image generation prompts based on the human and AI messages, including chat
    summary, past image prompts, and recent chat history.

    Args:
        chat_gen (BaseMessage): Response from AI prompt generator.

    Returns:
        str: The generated image generation prompt.
    """
    image_gen_prompts = []
    try:
        raw_prompts = storyboard.strip().split("\n")
        
        async for i in async_range(len(raw_prompts)):
            image_gen_prompt = raw_prompts[i].strip()
            # Replace with the second paragraph if the first one ends in a ':' character
            if image_gen_prompt.endswith(":"):
                continue
            # Remove leading and trailing punctuation
            image_gen_prompt = image_gen_prompt.rstrip(".,!?\"").lstrip("1234567890.,:\"")
            # Remove all quotes
            image_gen_prompt = image_gen_prompt.replace('"', '')
        
            # Apply the image generation prompt prefix
            prompt_components = []
            prefix = STORYBOARD_GENERATION_PROMPT_PREFIX
            if prefix.strip() != "":
                prompt_components.append(prefix)
            if image_gen_prompt != "":
                prompt_components.append(image_gen_prompt)
            postfix = STORYBOARD_GENERATION_PROMPT_POSTFIX
            if postfix.strip() != "":
                prompt_components.append(postfix)
        
            full_prompt = ", ".join(prompt_components)
            # Check refusal list
            if any(image_gen_prompt.startswith(refusal) for refusal in REFUSAL_LIST):
                cl_element.logger.warning(f"LLM refused to generate image prompt. Prompt is a refusal: {full_prompt}")
                raise ValueError("LLM refused to generate image prompt.")
        
            # Check for short prompts
            if len(image_gen_prompt) < 20 or not image_gen_prompt.strip():
                cl_element.logger.warning(f"Generated image prompt is too short or empty: {full_prompt}")
            else:
                image_gen_prompts.append(full_prompt)
    except Exception as e:
        cl_element.logger.error(f"Image prompt generation failed: {e}", exc_info=True)
        image_gen_prompts = []

    cl_element.logger.debug(f"Generated Image Generation Prompt: {image_gen_prompts}")

    return image_gen_prompts

@task
async def process_storyboard_images(storyboard: str, message_id: str) -> None:
    """Process storyboard into images and send to chat."""
    if not storyboard or not IMAGE_GENERATION_ENABLED:
        return
        
    try:
        # Generate image prompts
        image_prompts = await generate_image_generation_prompts(storyboard)
        
        # Process each prompt in order
        for prompt in image_prompts:
            try:
                # Generate image
                seed = random.randint(0, 2**32)
                image_bytes = await generate_image_async(prompt, seed)
                
                if image_bytes:
                    # Create and send image message
                    image_element = CLImage(
                        content=image_bytes,
                        display="inline",
                        size="large",
                        alt="Generated Image",
                        name="generated_image"
                    )
                    
                    await CLMessage(
                        content=f"**Image Generation Prompt:**\n{prompt}\n\n**Seed:**\n{seed}",
                        elements=[image_element],
                        parent_id=message_id
                    ).send()
                    
            except Exception as e:
                cl_element.logger.error(f"Failed to generate image for prompt: {prompt}. Error: {str(e)}")
                
    except Exception as e:
        cl_element.logger.error(f"Failed to process storyboard images: {str(e)}")

@on_chat_start
async def on_chat_start():
    """Initialize new chat session with Chainlit integration."""
    settings = await cl_element.ChatSettings(
        [
            Select(
                id="Model",
                label="OpenAI - Model",
                values=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-32k"],
                initial_index=0,
            )
        ]
    ).send()
    
    # Initialize user session
    user_id = context.session.user.id
    user_session = await DatabasePool.get_pool().get_user(user_id) or await DatabasePool.get_pool().create_user({"id": user_id, "name": context.session.user.name})
    cl_user_session.set("user_session", user_session)

    # Create initial state
    state = ChatState(
        messages=[SystemMessage(content=AI_WRITER_PROMPT)],
        thread_id=context.session.id,
        user_preferences=user_session.get("preferences", {}),
    )
    
    # Initialize thread in Chainlit
    await CLMessage(content=AI_WRITER_PROMPT, author="system").send()
    
    # Store state
    cl_user_session.set("state", state)
    cl_user_session.set("image_generation_memory", [])
    cl_user_session.set("ai_message_id", None)
    
    # Setup runnable
    cl_user_session.set("runnable", graph)
    
    # Load knowledge documents
    await load_knowledge_documents()
    
    # Send starters
    for starter in CHAINLIT_STARTERS:
        msg = await CLMessage(content=starter).send()
        state.messages.append(AIMessage(content=starter, additional_kwargs={"message_id": msg.id}))

@on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread."""
    # Set the user in the session
    user_dict = thread.get('user')
    if user_dict:
        cl_user_session.set('user', User(**user_dict))

    messages = [SystemMessage(content=AI_WRITER_PROMPT)]
    image_generation_memory = []
    
    # Reconstruct messages from thread history
    for step in sorted(thread.get("steps", []), key=lambda m: m.get("createdAt", "")):
        if step["type"] == "user_message":
            messages.append(HumanMessage(
                content=step["output"],
                additional_kwargs={"message_id": step["id"]}
            ))
        elif step["type"] == "ai_message":
            messages.append(AIMessage(
                content=step["output"],
                additional_kwargs={"message_id": step["id"]}
            ))
        elif step["type"] == "tool":  # Add handling for tool messages
            messages.append(ToolMessage(
                content=step["output"],
                tool_call_id=step.get("tool_call_id", f"restored_tool_{step['id']}"),
                name=step.get("name", "unknown_tool")
            ))
        elif step["type"] == "image_generation":
            image_generation_memory.append(step["output"])
    
    # Create state
    state = ChatState(
        messages=messages,
        thread_id=thread["id"],
        user_preferences=cl_user_session.get("user_session", {}).get("preferences", {}),
        thread_data=cl_user_session.get("thread_data", {})
    )
    
    # Store state and memories
    cl_user_session.set("state", state)
    cl_user_session.set("image_generation_memory", image_generation_memory)
    cl_user_session.set("ai_message_id", None)
    
    # Setup runnable
    cl_user_session.set("runnable", graph)
    
    # Load knowledge documents
    await load_knowledge_documents()

@on_message
async def on_message(message: CLMessage):
    """Handle incoming messages."""
    state = cl_user_session.get("state")
    runnable = cl_user_session.get("runnable")

    if message.type != "user_message":
        return

    config = {"configurable": {"thread_id": context.session.id}}
    cb = LangchainCallbackHandler()

    try:
        # Log the state before processing
        cl_element.logger.debug(f"Processing message: {message.content}")
        cl_element.logger.debug(f"Current state: {state}")
        
        # Add user message to state
        state = cl_user_session.get("state")
        state.messages.append(HumanMessage(content=message.content))
        
        # Handle dice roll if the message contains a dice roll command
        if "roll" in message.content.lower() and DICE_ROLLING_ENABLED:
            await handle_dice_roll(message)
            return

        # Format system message before generating response
        state.format_system_message()
        
        # Generate AI response
        ai_response = CLMessage(content="")
        async for chunk in runnable.astream(
            state,
            config=RunnableConfig(callbacks=[cb], **config)
        ):
            if isinstance(chunk, dict) and chunk.get("messages"):
                await ai_response.stream_token(chunk["messages"][-1].content)
        
        await ai_response.send()
        
        # Update state with the new message ID
        ai_message_id = ai_response.id
        state.metadata["current_message_id"] = ai_message_id
        
        # Handle image generation if there's a storyboard and image generation is enabled
        if "storyboard" in chunk and IMAGE_GENERATION_ENABLED:
            asyncio.create_task(process_storyboard_images(
                chunk["storyboard"],
                ai_message_id
            ))
        
        # Update session state
        cl_user_session.set("state", state)
    except Exception as e:
        cl_element.logger.error(f"Runnable stream failed: {e}", exc_info=True)
        cl_element.logger.error(f"State metadata: {state.metadata}")  # Log the state's metadata
        await CLMessage(content="⚠️ An error occurred while generating the response. Please try again later.").send()
        return

    # Update session state
    cl_user_session.set("state", state)

async def load_knowledge_documents():
    """
    Loads documents from the knowledge directory into the vector store.
    """
    if not os.path.exists(KNOWLEDGE_DIRECTORY):
        cl_element.logger.warning(f"Knowledge directory '{KNOWLEDGE_DIRECTORY}' does not exist. Skipping document loading.")
        return

    vector_memory = cl_user_session.get("vector_memory", None)

    if not vector_memory:
        cl_element.logger.error("Vector memory not initialized.")
        return

    documents = []

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for root, dirs, files in os.walk(KNOWLEDGE_DIRECTORY):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
            elif file.endswith(".txt"):
                loader = TextLoader(file_path)
            elif file.endswith(".md"):
                loader = UnstructuredMarkdownLoader(file_path)
            else:
                cl_element.logger.warning(f"Unsupported file type: {file}. Skipping.")
                continue
            try:
                loaded_docs = loader.load()
                split_docs = text_splitter.split_documents(loaded_docs)
                documents.extend(split_docs)
            except Exception as e:
                cl_element.logger.error(f"Error loading document {file_path}: {e}")

    if documents:
        cl_element.logger.info(f"Adding {len(documents)} documents to the vector store.")
        vector_memory.retriever.vectorstore.add_documents(documents)
    else:
        cl_element.logger.info("No documents found to add to the vector store.")
