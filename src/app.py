import os

from initialization import *
from tools_and_agents import *
from state_graph import graph
from memory_management import *
from image_generation import *
from event_handlers import *

import logging

# Initialize logging
cl_logger = logging.getLogger("chainlit")

def setup_runnable():
    """
    Sets up the runnable for generating AI responses based on user input and conversation history.
    """
    cl_logger.info("Setting up runnable for generating AI responses.")
    cl.user_session.set("runnable", graph)

@cl.password_auth_callback
def auth_callback(username: str, password: str) -> Optional[cl.User]:
    """
    Authentication callback for Chainlit. Authenticates user with username and password.

    Returns:
        Optional[cl.User]: The authenticated user or None if authentication fails.
    """
    if (username, password) == ("admin", "admin"):
        cl_logger.info(f"Authentication successful for user: {username}")
        return cl.User(
            id="admin_id",  # Set a unique id
            identifier="admin",
            # Ensure metadata is a dictionary
            metadata={"role": "admin", "provider": "credentials"}
        )
    if (username, password) == ("guest", "guest"):
        cl_logger.info(f"Authentication successful for user: {username}")
        return cl.User(
            id="guest_id",  # Set a unique id
            identifier="guest",
            # Ensure metadata is a dictionary
            metadata={"role": "guest", "provider": "credentials"}
        )
    else:
        cl_logger.warning("Authentication failed for user: %s", username)
        return None
