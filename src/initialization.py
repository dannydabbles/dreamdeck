import os
import logging
from chainlit.data.base import BaseDataLayer
from .data_layer import CustomDataLayer

# Initialize logging
cl_logger = logging.getLogger("chainlit")

class DatabasePool:
    _instance: Optional[BaseDataLayer] = None
    _initialized: bool = False

    @classmethod
    async def get_pool(cls) -> BaseDataLayer:
        if not cls._initialized:
            await cls.initialize()
        return cls._instance  # type: ignore

    @classmethod
    async def initialize(cls) -> None:
        if not cls._initialized:
            # Initialize the custom data layer
            cls._instance = CustomDataLayer(
                database_url=os.getenv("DATABASE_URL"),
                storage_client=None
            )
            cls._initialized = True

    @classmethod
    async def close(cls) -> None:
        if cls._instance:
            await cls._instance.close()
            cls._instance = None
            cls._initialized = False

# Initialize the custom data layer
custom_data_layer = CustomDataLayer()

# Initialize the database pool
async def init_db():
    await DatabasePool.initialize()

# Initialize the custom data layer
async def init_data_layer():
    await custom_data_layer.initialize()
