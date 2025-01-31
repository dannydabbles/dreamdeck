import base64
import random
import asyncio
from typing import List, Optional, Sequence
import httpx
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langchain_core.messages.base import BaseMessage

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

import chainlit as cl

from config import (
    DENOISING_STRENGTH,
    DISTILLED_CFG_SCALE,
    HR_SECOND_PASS_STEPS,
    HR_UPSCALER,
    STORYBOARD_GENERATION_PROMPT,
    STORYBOARD_GENERATION_PROMPT_POSTFIX,
    STORYBOARD_GENERATION_PROMPT_PREFIX,
    LLM_FREQUENCY_PENALTY,
    LLM_MODEL_NAME,
    LLM_PRESENCE_PENALTY,
    LLM_TOP_P,
    LLM_TOP_K,
    LLM_VERBOSE,
    NEGATIVE_PROMPT,
    STEPS,
    SAMPLER_NAME,
    SCHEDULER,
    CFG_SCALE,
    WIDTH,
    HEIGHT,
    IMAGE_GENERATION_TIMEOUT,
    REFUSAL_LIST,
    STABLE_DIFFUSION_API_URL,
    LLM_TEMPERATURE,
    
)

from chainlit import Message as CLMessage
from chainlit.element import Image as CLImage

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

    # Stable diffsuion payload
    # payload = {
    #     "prompt": image_generation_prompt,
    #     "negative_prompt": NEGATIVE_PROMPT,
    #     "steps": STEPS,
    #     "sampler_name": SAMPLER_NAME,
    #     "scheduler": SCHEDULER                            ,
    #     "cfg_scale": CFG_SCALE,
    #     "width": WIDTH,
    #     "height": HEIGHT,
    #     "seed": seed,
    # }

    try:
        async with httpx.AsyncClient(timeout=IMAGE_GENERATION_TIMEOUT) as client:
            response = await client.post(f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img", json=payload)
            response.raise_for_status()
            image_data = response.json()["images"][0]
            image_bytes = base64.b64decode(image_data)
            return image_bytes
    except httpx.RequestError as e:
        cl.logger.error(f"Image generation failed: {e}")
        raise
    except (KeyError, IndexError, ValueError) as e:
        cl.logger.error(f"Error processing image data: {e}")
        return None

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
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
                cl.logger.warning(f"LLM refused to generate image prompt. Prompt is a refusal: {full_prompt}")
                raise ValueError("LLM refused to generate image prompt.")
        
            # Check for short prompts
            if len(image_gen_prompt) < 20 or not image_gen_prompt.strip():
                cl.logger.warning(f"Generated image prompt is too short or empty: {full_prompt}")
            else:
                image_gen_prompts.append(full_prompt)
    except Exception as e:
        cl.logger.error(f"Image prompt generation failed: {e}")
        image_gen_prompts = []

    cl.logger.debug(f"Generated Image Generation Prompt: {image_gen_prompts}")

    return image_gen_prompts

async def handle_image_generation(image_generation_prompts: List[str], parent_id: str):
    """
    Handles the asynchronous image generation process and sends the image once it's ready.

    Args:
        image_generation_prompts (List[str]): The image generation prompt.
        parent_id (str): The message ID of the AI response to associate the image with.
    """
    async for i in async_range(len(image_generation_prompts)):
        image_generation_prompt = image_generation_prompts[i]
        try:
            seed = random.randint(0, 2**32)
            image_bytes = await generate_image_async(image_generation_prompt, seed)
        except Exception as e:
            cl.logger.error(f"Image generation failed after retries: {e}")
            image_bytes = None
            seed = None

        if image_bytes and seed:
            # Save the image generation prompt in image generation memory
            image_generation_memory = cl.user_session.get("image_generation_memory", [])
            image_generation_memory.append(image_generation_prompt)
            # Keep only the last 50 image prompts
            image_generation_memory = image_generation_memory[-50:]
            cl.user_session.set("image_generation_memory", image_generation_memory)

            # Send the image as a child message of the AI response
            image_element = CLImage(
                content=image_bytes,
                display="inline",
                size="large",
                alt="Generated Image",
                name="generated_image"
            )
            await CLMessage(
                    content=f"**Image Generation Prompt:**\n{image_generation_prompt}\n\n**Negative Prompt**:\n{NEGATIVE_PROMPT}\n\n**Seed:**\n{seed}",
                elements=[image_element],
                parent_id=parent_id  # Associate with the AI response
            ).send()
        else:
            cl.logger.warning("Image generation failed. Skipping image display.")
            await CLMessage(content="⚠️ Image generation failed. Please try again later.", parent_id=parent_id).send()
