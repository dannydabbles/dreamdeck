import base64
import random
import asyncio
from typing import List, Optional
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)
from chainlit import element as cl_element  # Import cl_element
from chainlit import Message as CLMessage  # Import CLMessage from Chainlit
from chainlit.types import CLImage  # Import CLImage
from .config import (
    DENOISING_STRENGTH,
    CFG_SCALE,
    HR_SECOND_PASS_STEPS,
    HR_UPSCALER,
    NEGATIVE_PROMPT,
    STEPS,
    SAMPLER_NAME,
    SCHEDULER,
    WIDTH,
    HEIGHT,
    IMAGE_GENERATION_TIMEOUT,
    REFUSAL_LIST,
    STABLE_DIFFUSION_API_URL,
    IMAGE_GENERATION_ENABLED,
    STORYBOARD_GENERATION_PROMPT_PREFIX,
    STORYBOARD_GENERATION_PROMPT_POSTFIX,
)
import httpx  # Import httpx
from .tools_and_agents import task  # Import task from tools_and_agents


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


@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
)
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

    try:
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

        async with httpx.AsyncClient(timeout=IMAGE_GENERATION_TIMEOUT) as client:
            response = await client.post(
                f"{STABLE_DIFFUSION_API_URL}/sdapi/v1/txt2img", json=payload
            )
            response.raise_for_status()
            image_data = response.json()["images"][0]
            image_bytes = base64.b64decode(image_data)
            return image_bytes

    except httpx.RequestError as e:
        cl_element.logger.error(f"Image generation failed after retries: {e}", exc_info=True)
        raise
    except (KeyError, IndexError, ValueError) as e:
        cl_element.logger.error(f"Error processing image data: {e}", exc_info=True)
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


@task
async def process_storyboard_images(storyboard: str, message_id: str) -> None:
    """Process storyboard into images and send to chat.

    Args:
        storyboard (str): The storyboard content.
        message_id (str): The message ID for the chat.
    """
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
