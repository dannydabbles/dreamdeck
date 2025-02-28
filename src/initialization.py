import os
import logging
from chainlit.data.base import ChainlitDataLayer
from typing import Optional  # Import Optional here

# Initialize logging
cl_logger = logging.getLogger("chainlit")


class DatabasePool:
    """Database pool manager for ChainlitDataLayer.

    Attributes:
        _instance (Optional[ChainlitDataLayer]): Singleton instance of the database pool.
        _initialized (bool): Flag indicating if the pool is initialized.
    """

    _instance: Optional[ChainlitDataLayer] = None
    _initialized: bool = False

    @classmethod
    async def get_pool(cls) -> ChainlitDataLayer:
        """Get the initialized database pool.

        Returns:
            ChainlitDataLayer: The database pool instance.
        """
        if not cls._initialized:
            await cls.initialize()
        return cls._instance  # type: ignore

    @classmethod
    async def initialize(cls) -> None:
        """Initialize the database pool if not already initialized."""
        if not cls._initialized:
            # Initialize the Chainlit data layer
            cls._instance = ChainlitDataLayer(
                database_url=os.getenv("DATABASE_URL"), storage_client=None
            )
            cls._initialized = True

    @classmethod
    async def close(cls) -> None:
        """Close the database pool and reset the instance."""
        if cls._instance:
            await cls._instance.close()
            cls._instance = None
            cls._initialized = False


# Initialize the database pool
async def init_db():
    """Initialize the database pool."""
    await DatabasePool.initialize()
