import os
import logging
import asyncio
from asyncpg.pool import Pool

# Initialize logging
cl_logger = logging.getLogger("chainlit")

class DatabasePool:
    _instance: Optional[Pool] = None
    _initialized: bool = False

    @classmethod
    async def get_pool(cls) -> Pool:
        if not cls._initialized:
            await cls.initialize()
        return cls._instance  # type: ignore

    @classmethod
    async def initialize(cls) -> None:
        if not cls._initialized:
            # Configure the database pool with proper settings
            cls._instance = await Pool.create(
                min_size=1,
                max_size=10,
                host=os.getenv("DATABASE_HOST", "localhost"),
                port=int(os.getenv("DATABASE_PORT", 5432)),
                user=os.getenv("DATABASE_USER", "postgres"),
                password=os.getenv("DATABASE_PASSWORD", "postgres"),
                database=os.getenv("DATABASE_NAME", "chainlit"),
                max_queries=1000,
                max_inactive=300,
            )
            cls._initialized = True

    @classmethod
    async def close(cls) -> None:
        if cls._instance:
            await cls._instance.close()
            cls._instance = None
            cls._initialized = False

# Initialize the database pool
async def init_db():
    await DatabasePool.initialize()

# Initialize the storage client
# fs_storage_client = FSStorageClient(
#     storage_path=os.path.join(os.getcwd(), "public", "storage"),
#     url_path=os.path.join("public", "storage")
# )

# Mount the static files directory to serve images
# chainlit_app.mount("/storage", StaticFiles(directory=os.path.join("public", "storage")), name="storage")

# Initialize the custom data layer with the storage client
# cl_data._data_layer = CustomDataLayer(conninfo=DATABASE_URL, storage_provider=[fs_storage_client])
