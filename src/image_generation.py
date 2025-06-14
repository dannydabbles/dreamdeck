import asyncio
import base64
import random
from typing import List, Optional

import httpx  # Import httpx
from chainlit import Image as CLImage  # Import Image from Chainlit
from chainlit import Message as CLMessage  # Import CLMessage from Chainlit
from chainlit import element as cl_element  # Import cl_element
from langgraph.func import task  # Import task from langgraph
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    CFG_SCALE,
    DENOISING_STRENGTH,
    HEIGHT,
    HR_SECOND_PASS_STEPS,
    HR_UPSCALER,
    IMAGE_GENERATION_ENABLED,
    IMAGE_GENERATION_TIMEOUT,
    KNOWLEDGE_DIRECTORY,
    NEGATIVE_PROMPT,
    REFUSAL_LIST,
    SAMPLER_NAME,
    SCHEDULER,
    STABLE_DIFFUSION_API_URL,
    STEPS,
    STORYBOARD_GENERATION_PROMPT_POSTFIX,
    STORYBOARD_GENERATION_PROMPT_PREFIX,
    WIDTH,
)

import logging

cl_logger = logging.getLogger("chainlit")

# --- API Connectivity Check on Startup ---
import chainlit as cl


@cl.on_chat_start
async def check_sd_api():
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/options")
            cl_logger.info(f"Stable Diffusion API connected: {resp.status_code}")
    except Exception as e:
        cl_logger.error(f"Stable Diffusion connection failed: {str(e)}")
        await cl.Message(
            content="⚠️ Image generation unavailable: API connection failed"
        ).send()


# Define an asynchronous range generator
async def async_range(end):
    for i in range(end):
        await asyncio.sleep(0)  # Simulate async
        yield i  # Use yield to create async generator


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
)
async def generate_image_async(
    image_generation_prompt: str, seed: int
) -> Optional[bytes]:
    if not IMAGE_GENERATION_ENABLED:
        cl_logger.warning("Image generation disabled by config")
        return None

    try:
        cl_logger.info(f"Attempting connection to {STABLE_DIFFUSION_API_URL}")
        async with httpx.AsyncClient(timeout=IMAGE_GENERATION_TIMEOUT) as client:
            # First check API availability
            try:
                health_check = await client.get(
                    f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/options"
                )
                health_check.raise_for_status()
                cl_logger.info(
                    f"API connection successful ({health_check.status_code})"
                )
            except Exception as health_error:
                cl_logger.error(
                    f"Stable Diffusion API unavailable: {str(health_error)}"
                )
                return None

            # Proceed with image generation if health check passed
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

            response = await client.post(
                f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img", json=payload
            )
            response.raise_for_status()
            image_data = response.json()["images"][0]
            image_bytes = base64.b64decode(image_data)
            return image_bytes

    except httpx.RequestError as e:
        cl_logger.error(f"Image generation failed after retries: {e}", exc_info=True)
        raise
    except (KeyError, IndexError, ValueError) as e:
        cl_logger.error(f"Error processing image data: {e}", exc_info=True)
        return None


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
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


async def process_storyboard_images(
    storyboard: str, message_id: str, sd_api_url: str = None
) -> None:
    """Process storyboard into images and send to chat.

    Args:
        storyboard (str): The storyboard content.
        message_id (str): The message ID for the chat.
        sd_api_url (str): The Stable Diffusion API URL to use.
    """
    cl_logger.info(f"Starting image generation with API: {sd_api_url}")
    if not storyboard or not IMAGE_GENERATION_ENABLED:
        cl_logger.warning("Image generation skipped - no content or disabled")
        return

    try:
        image_prompts = await generate_image_generation_prompts(storyboard)
        cl_logger.info(f"Generated {len(image_prompts)} image prompts")

        for prompt in image_prompts:
            cl_logger.debug(f"Processing prompt: {prompt[:60]}...")
            try:
                # Health check with explicit URL
                if sd_api_url:
                    async with httpx.AsyncClient() as client:
                        health_resp = await client.get(f"{sd_api_url}/sdapi/v1/options")
                        health_resp.raise_for_status()  # Now triggers retry
                        cl_logger.info(
                            f"API health check status: {health_resp.status_code}"
                        )
                        if health_resp.status_code != 200:
                            await cl.Message(
                                content="⚠️ Image generation service unavailable",
                                parent_id=message_id,
                            ).send()
                            return

                # Generate image
                seed = random.randint(0, 2**32)
                # Pass sd_api_url to generate_image_async if supported
                import inspect

                if (
                    "sd_api_url" in inspect.signature(generate_image_async).parameters
                    and sd_api_url
                ):
                    image_bytes = await generate_image_async(
                        prompt, seed, sd_api_url=sd_api_url
                    )
                else:
                    image_bytes = await generate_image_async(prompt, seed)

                if image_bytes:
                    # Create and send image message
                    image_element = CLImage(
                        content=image_bytes,
                        display="inline",
                        size="large",
                        alt="Generated Image",
                        name="generated_image",
                    )

                    await CLMessage(
                        content=f"**Image Generation Prompt:**\n{prompt}\n\n**Seed:**\n{seed}",
                        elements=[image_element],
                        parent_id=message_id,
                    ).send()

            except Exception as e:
                cl_element.logger.error(
                    f"Failed to generate image for prompt: {prompt}. Error: {str(e)}"
                )

    except Exception as e:
        cl_element.logger.error(f"Failed to process storyboard images: {str(e)}")
