import os
import asyncio
import random
import base64
import httpx
from typing import List, Optional, Dict as ThreadDict
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from chainlit import (
    on_chat_start,
    on_chat_resume,
    on_message,
    user_session as cl_user_session,
    Message as CLMessage,
    element as cl_element,
)
from chainlit.types import SystemMessage, HumanMessage, AIMessage, ToolMessage
from chainlit import LangchainCallbackHandler
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnableConfig  # Import RunnableConfig
from .config import (
    NEGATIVE_PROMPT,
    STEPS,
    SAMPLER_NAME,
    SCHEDULER,
    CFG_SCALE,
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
    MAX_RETRIES,
    RETRY_DELAY,
    IMAGE_GENERATION_ENABLED,
    WEB_SEARCH_ENABLED,
    DICE_ROLLING_ENABLED,
    STABLE_DIFFUSION_API_ENDPOINT,
)
from .state import ChatState
from .state_graph import chat_workflow
from .tools_and_agents import handle_dice_roll  # Import handle_dice_roll
from .initialization import DatabasePool  # Import DatabasePool
from .image_generation import process_storyboard_images


# Define an asynchronous range generator
async def async_range(end):
    """Asynchronous range generator.

    Args:
        end (int): The end value for the range.
    """
    for i in range(0, end):
        # Sleep for a short duration to simulate asynchronous operation
        await asyncio.sleep(0.1)
        yield i


async def generate_image_async(
    image_generation_prompt: str, seed: int
) -> Optional[bytes]:
    """Generate an image asynchronously using the Stable Diffusion API.

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
        "width": WIDTH,
        "height": HEIGHT,
        "hr_upscaler": HR_UPSCALER,
        "denoising_strength": DENOISING_STRENGTH,
        "hr_second_pass_steps": HR_SECOND_PASS_STEPS,
        "seed": seed,
    }

    try:
        async with httpx.AsyncClient(timeout=IMAGE_GENERATION_TIMEOUT) as client:
            response = await client.post(
                f"{STABLE_DIFFUSION_API_ENDPOINT}/sdapi/v1/txt2img", json=payload
            )
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
    retry=retry_if_exception_type(httpx.RequestError),
)
async def generate_image_generation_prompts(storyboard: str) -> List[str]:
    """Generate a list of image generation prompts based on the storyboard.

    Args:
        storyboard (str): The storyboard content.

    Returns:
        List[str]: List of generated image generation prompts.
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
            image_gen_prompt = image_gen_prompt.rstrip('.,!?"').lstrip('1234567890.,:"')
            # Remove all quotes
            image_gen_prompt = image_gen_prompt.replace('"', "")

            # Apply the image generation prompt prefix
            prompt_components = []
            if STORYBOARD_GENERATION_PROMPT_PREFIX.strip() != "":
                prompt_components.append(STORYBOARD_GENERATION_PROMPT_PREFIX)
            if image_gen_prompt != "":
                prompt_components.append(image_gen_prompt)
            if STORYBOARD_GENERATION_PROMPT_POSTFIX.strip() != "":
                prompt_components.append(STORYBOARD_GENERATION_PROMPT_POSTFIX)

            full_prompt = ", ".join(prompt_components)
            # Check refusal list
            if any(image_gen_prompt.startswith(refusal) for refusal in REFUSAL_LIST):
                cl_element.logger.warning(
                    f"LLM refused to generate image prompt. Prompt is a refusal: {full_prompt}"
                )
                raise ValueError("LLM refused to generate image prompt.")

            # Check for short prompts
            if len(image_gen_prompt) < 20 or not image_gen_prompt.strip():
                cl_element.logger.warning(
                    f"Generated image prompt is too short or empty: {full_prompt}"
                )
            else:
                image_gen_prompts.append(full_prompt)
    except Exception as e:
        cl_element.logger.error(f"Image prompt generation failed: {e}", exc_info=True)
        image_gen_prompts = []

    cl_element.logger.debug(f"Generated Image Generation Prompt: {image_gen_prompts}")

    return image_gen_prompts


@on_chat_start
async def on_chat_start():
    """Initialize new chat session with Chainlit integration.

    Sets up the user session, initializes the chat state, and sends initial messages.
    """
    # Initialize user session
    user_id = cl_user_session.context.session.user.id
    user_session = await DatabasePool.get_pool().get_user(
        user_id
    ) or await DatabasePool.get_pool().create_user(
        {"id": user_id, "name": cl_user_session.context.session.user.name}
    )
    cl_user_session.set("user_session", user_session)

    # Create initial state
    state = ChatState(
        messages=[CLMessage(content=AI_WRITER_PROMPT, type="system")],
        thread_id=cl_user_session.context.session.id,
        user_preferences=user_session.get("preferences", {}),
    )

    # Initialize thread in Chainlit
    await CLMessage(content=AI_WRITER_PROMPT, author="system").send()

    # Store state
    cl_user_session.set("state", state)
    cl_user_session.set("image_generation_memory", [])
    cl_user_session.set("ai_message_id", None)

    # Setup runnable
    cl_user_session.set("runnable", chat_workflow)

    # Load knowledge documents
    await load_knowledge_documents()


@on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Reconstruct state from Chainlit thread.

    Args:
        thread (ThreadDict): The thread dictionary from Chainlit.
    """
    # Set the user in the session
    user_dict = thread.get("user")
    if user_dict:
        cl_user_session.set("user", user_dict)

    messages = [CLMessage(content=AI_WRITER_PROMPT, type="system")]
    image_generation_memory = []

    # Reconstruct messages from thread history
    for step in sorted(thread.get("steps", []), key=lambda m: m.get("createdAt", "")):
        if step["type"] == "user_message":
            messages.append(
                CLMessage(
                    content=step["output"], additional_kwargs={"message_id": step["id"]}, type="user"
                )
            )
        elif step["type"] == "ai_message":
            messages.append(
                CLMessage(
                    content=step["output"], additional_kwargs={"message_id": step["id"]}, type="ai"
                )
            )
        elif step["type"] == "tool":  # Add handling for tool messages
            messages.append(
                CLMessage(
                    content=step["output"],
                    tool_call_id=step.get("tool_call_id", f"restored_tool_{step['id']}"),
                    name=step.get("name", "unknown_tool"),
                    type="tool",
                )
            )
        elif step["type"] == "image_generation":
            image_generation_memory.append(step["output"])

    # Create state
    state = ChatState(
        messages=messages,
        thread_id=thread["id"],
        user_preferences=cl_user_session.get("user_session", {}).get("preferences", {}),
        thread_data=cl_user_session.get("thread_data", {}),
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
    """Handle incoming messages.

    Args:
        message (CLMessage): The incoming message.
    """
    state = cl_user_session.get("state")
    runnable = cl_user_session.get("runnable")

    if message.type != "user_message":
        return

    config = {"configurable": {"thread_id": cl_user_session.context.session.id}}
    cb = LangchainCallbackHandler()

    try:
        # Log the state before processing
        cl_element.logger.debug(f"Processing message: {message.content}")
        cl_element.logger.debug(f"Current state: {state}")

        # Add user message to state
        state.messages.append(CLMessage(content=message.content, type="user"))

        # Format system message before generating response
        state.format_system_message()

        # Generate AI response using the chat workflow
        state = await chat_workflow(state.messages, store=cl_user_session.get("vector_memory"), previous=state)

        # Update session state
        cl_user_session.set("state", state)

    except Exception as e:
        cl_element.logger.error(f"Runnable stream failed: {e}", exc_info=True)
        cl_element.logger.error(
            f"State metadata: {state.metadata}"
        )  # Log the state's metadata
        await CLMessage(
            content="⚠️ An error occurred while generating the response. Please try again later."
        ).send()
        return

    # Update session state
    cl_user_session.set("state", state)


async def load_knowledge_documents():
    """Load documents from the knowledge directory into the vector store."""
    if not os.path.exists(KNOWLEDGE_DIRECTORY):
        cl_element.logger.warning(
            f"Knowledge directory '{KNOWLEDGE_DIRECTORY}' does not exist. Skipping document loading."
        )
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
        cl_element.logger.info(
            f"Adding {len(documents)} documents to the vector store."
        )
        vector_memory.retriever.vectorstore.add_documents(documents)
    else:
        cl_element.logger.info("No documents found to add to the vector store.")
