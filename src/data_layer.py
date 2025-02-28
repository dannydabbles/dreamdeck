from chainlit.data.base import BaseDataLayer
from typing import Dict, Any, Optional
import logging
from .initialization import DatabasePool

cl_logger = logging.getLogger("chainlit")

class CustomDataLayer(BaseDataLayer):
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user data from the database."""
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
            return dict(result) if result else None

    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user in the database."""
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                "INSERT INTO users (id, name, preferences) VALUES ($1, $2, $3) RETURNING *",
                user_data["id"], user_data["name"], user_data.get("preferences", {})
            )
            return dict(result)

    async def upsert_feedback(self, feedback: Dict[str, Any]) -> None:
        """Update or insert feedback data."""
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO feedback (user_id, feedback, timestamp) VALUES ($1, $2, $3) ON CONFLICT (user_id) DO UPDATE SET feedback = EXCLUDED.feedback, timestamp = EXCLUDED.timestamp",
                feedback["user_id"], feedback["feedback"], feedback["timestamp"]
            )

    async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve thread data from the database."""
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow("SELECT * FROM threads WHERE id = $1", thread_id)
            return dict(result) if result else None

    async def create_thread(self, thread_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new thread in the database."""
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            result = await conn.fetchrow(
                "INSERT INTO threads (id, user_id, messages) VALUES ($1, $2, $3) RETURNING *",
                thread_data["id"], thread_data["user_id"], thread_data.get("messages", [])
            )
            return dict(result)

    async def update_thread(self, thread_id: str, thread_data: Dict[str, Any]) -> None:
        """Update an existing thread in the database."""
        pool = await DatabasePool.get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE threads SET messages = $1 WHERE id = $2",
                thread_data["messages"], thread_id
            )

    async def get_user_session(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user session data."""
        user_data = await self.get_user(user_id)
        if user_data:
            return user_data.get("session_data", {})
        return {}

    async def save_user_session(self, user_id: str, session_data: Dict[str, Any]) -> None:
        """Save user session data."""
        user_data = await self.get_user(user_id)
        if user_data:
            user_data["session_data"] = session_data
            await self.update_user(user_data)
