from chainlit.data.base import BaseDataLayer
from chainlit.types import ThreadDict, Feedback
from typing import Dict, Any, Optional, List  # Import List here
import logging

cl_logger = logging.getLogger("chainlit")

class CustomDataLayer(BaseDataLayer):
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user data from the database."""
        return await super().get_user(user_id)

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user in the database."""
        return await super().create_user(user_data)

    async def upsert_feedback(self, feedback: Feedback) -> None:
        """Update or insert feedback data."""
        await super().upsert_feedback(feedback)

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        """Retrieve thread data from the database."""
        return await super().get_thread(thread_id)

    async def create_thread(self, thread_data: Dict[str, Any]) -> ThreadDict:
        """Create a new thread in the database."""
        return await super().create_thread(thread_data)

    async def update_thread(self, thread_id: str, thread_data: Dict[str, Any]) -> None:
        """Update an existing thread in the database."""
        await super().update_thread(thread_id, thread_data)

    async def get_user_session(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user session data."""
        return await super().get_user_session(user_id)

    async def save_user_session(self, user_id: str, session_data: Dict[str, Any]) -> None:
        """Save user session data."""
        await super().save_user_session(user_id, session_data)

    async def list_threads(self, user_id: str) -> List[ThreadDict]:
        """List all threads for a user."""
        return await super().list_threads(user_id)

    async def delete_thread(self, thread_id: str) -> None:
        """Delete a thread."""
        await super().delete_thread(thread_id)

    async def delete_feedback(self, feedback_id: str) -> None:
        """Delete feedback."""
        await super().delete_feedback(feedback_id)
