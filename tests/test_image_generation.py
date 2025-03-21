import pytest
from unittest.mock import patch, AsyncMock  # <-- ADD THIS IMPORT
from src.image_generation import generate_image_async, generate_image_generation_prompts
import base64  # Import base64

@pytest.mark.asyncio
async def test_image_prompt_generation():
    with patch("src.image_generation.async_range", new=AsyncMock(return_value=[])):  # Mock async gen
        prompts = await generate_image_generation_prompts("Scene description\nNext scene")
        assert len(prompts) == 2

@pytest.mark.asyncio
async def test_mocked_image_generation():
    with patch("httpx.AsyncClient.post") as mock_post:  # Fix module path
        mock_post.return_value.json.return_value = {"images": ["dummy_base64"]}
        image_bytes = await generate_image_async("Test prompt", 123)
        assert image_bytes == base64.b64decode("dummy_base64")
