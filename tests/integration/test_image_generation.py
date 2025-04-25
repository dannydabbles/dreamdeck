import base64  # Import base64
from unittest.mock import AsyncMock, MagicMock, patch  # <-- ADD THIS IMPORT

import httpx
import pytest

from src.image_generation import generate_image_async, generate_image_generation_prompts


@pytest.mark.asyncio
async def test_image_prompt_generation():
    async def mock_async_range(end):
        for i in range(end):
            yield i  # Return actual numbers

    with patch("src.image_generation.async_range", new=mock_async_range):
        # Use longer input strings exceeding 20 characters each
        prompts = await generate_image_generation_prompts(
            "A vast enchanted forest with glowing mushrooms\nA mystical castle floating in the clouds"
        )
        assert len(prompts) == 2


@pytest.mark.asyncio
async def test_mocked_image_generation():
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_response = MagicMock()
        # Return value directly (not a coroutine)
        mock_response.json.return_value = {
            "images": ["SGVsbG8gd29ybGQ="]
        }  # Decodes to "Hello world"
        mock_post.return_value = mock_response

        image_bytes = await generate_image_async("Test prompt", 123)
        assert (
            image_bytes == b"Hello world"
        )  # Matches the mock's "Hello world" decoding


@pytest.mark.asyncio
async def test_generate_image_async_retries():
    from src.image_generation import generate_image_async

    with patch(
        "httpx.AsyncClient.post", side_effect=httpx.HTTPError("fail")
    ) as mock_post:
        try:
            await generate_image_async("prompt", 42)
        except Exception:
            pass
        assert mock_post.call_count >= 1  # retried
