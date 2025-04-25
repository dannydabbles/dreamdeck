import json
import os
import zoneinfo
from datetime import datetime
from pathlib import Path

from src.models import ChatState

try:
    # Try to use tzlocal if available for system local time zone
    import tzlocal

    LOCAL_TZ = tzlocal.get_localzone()
except ImportError:
    # Fallback to system local time zone via zoneinfo
    try:
        LOCAL_TZ = zoneinfo.ZoneInfo(datetime.now().astimezone().tzinfo.key)
    except Exception:
        # Fallback to UTC if all else fails
        LOCAL_TZ = zoneinfo.ZoneInfo("UTC")


def save_state(state: ChatState, path: str):
    """Save ChatState to a JSON file."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(path_obj, "w", encoding="utf-8") as f:
        json.dump(state.model_dump(), f, indent=2)


def load_state(path: str) -> ChatState:
    """Load ChatState from a JSON file."""
    path_obj = Path(path)
    if not path_obj.exists():
        # Return empty state if file missing
        return ChatState(messages=[], thread_id="cli_thread")
    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)
    return ChatState.model_validate(data)


def get_local_now():
    """Get the current datetime in the system's local timezone."""
    return datetime.now(LOCAL_TZ)


def get_shared_daily_dir(date: str = None) -> Path:
    """Return the shared daily directory path."""
    if date is None:
        date = get_local_now().strftime("%Y-%m-%d")
    return Path("helper") / "shared" / date


def get_persona_daily_dir(persona: str, date: str = None) -> Path:
    """Return the persona-specific daily directory path."""
    if date is None:
        date = get_local_now().strftime("%Y-%m-%d")
    safe_persona = "".join(c if c.isalnum() or c in "-_." else "_" for c in persona)
    return Path("helper") / safe_persona / date


def ensure_dir(path: Path):
    """Ensure a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def save_text_file(path: Path, content: str):
    """Save plain text content to a file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def load_text_file(path: Path) -> str:
    """Load plain text content from a file."""
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def append_log(persona: str, message: str, date: str = None):
    """Append a log entry for a persona on a given date."""
    if date is None:
        date = get_local_now().strftime("%Y-%m-%d")
    safe_persona = "".join(c if c.isalnum() or c in "-_." else "_" for c in persona)
    log_dir = Path("helper") / safe_persona / date
    log_file = log_dir / "log.txt"
    ensure_dir(log_dir)
    timestamp = get_local_now().isoformat()
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
