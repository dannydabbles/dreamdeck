import os
import logging
from chainlit import cl
from chainlit.server import app as chainlit_app
from src.initialization import init_db, DatabasePool

# Initialize logging
cl_logger = logging.getLogger("chainlit")

async def main():
    try:
        # Initialize database pool
        await init_db()
        
        # Run Chainlit server
        await chainlit_app.run()
    finally:
        # Close database pool
        await DatabasePool.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
