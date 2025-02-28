import os
import logging
from chainlit import cl
from chainlit.server import app as chainlit_app
from src.initialization import init_db, DatabasePool
from src.stores import VectorStore  # Import VectorStore

# Centralized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chainlit.log')
    ]
)

cl_logger = logging.getLogger("chainlit")

async def main():
    """Main entry point for the application.
    
    Initializes the database pool, vector store, and runs the Chainlit server.
    """
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
