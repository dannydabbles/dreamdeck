import os
import logging
from chainlit.server import app as chainlit_app
from src.initialization import init_db, DatabasePool
from src.stores import VectorStore  # Import VectorStore
from src.agents.decision_agent import decision_agent
from src.agents.writer_agent import writer_agent
from src.agents.storyboard_editor_agent import storyboard_editor_agent
from src.agents.dice_agent import dice_roll_agent
from src.agents.web_search_agent import web_search_agent

# Centralized logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("chainlit.log")],
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

        # Initialize agents
        cl_user_session.set("decision_agent", decision_agent)
        cl_user_session.set("writer_agent", writer_agent)
        cl_user_session.set("storyboard_editor_agent", storyboard_editor_agent)
        cl_user_session.set("dice_roll_agent", dice_roll_agent)
        cl_user_session.set("web_search_agent", web_search_agent)

        # Run Chainlit server
        await chainlit_app.run()
    except Exception as e:
        cl_logger.error(f"Application failed to start: {e}", exc_info=True)
        raise
    finally:
        # Close database pool
        await DatabasePool.close()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
