import os
import logging
from chainlit import cl
from chainlit.server import app as chainlit_app
from src.initialization import init_db, DatabasePool
from src.stores import VectorStore

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def main():
    try:
        # Initialize database pool
        await init_db()
        
        # Initialize vector store
        vector_memory = VectorStore()
        
        # Run Chainlit server
        await chainlit_app.run()
    finally:
        # Close database pool
        await DatabasePool.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
