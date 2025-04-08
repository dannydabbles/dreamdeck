import pytest
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_database_pool_lifecycle():
    from src.initialization import DatabasePool
    dummy_pool = MagicMock()
    dummy_pool.close = AsyncMock()
    with patch("src.initialization.ChainlitDataLayer", return_value=dummy_pool):
        await DatabasePool.initialize()
        pool = await DatabasePool.get_pool()
        assert pool is dummy_pool
        await DatabasePool.close()
