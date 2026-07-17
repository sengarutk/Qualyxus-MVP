"""Database logger compatibility shim.

The runtime event logger is implemented as `src.logging.event_logger.EventLogger`.
Some architectural docs refer to `src.database.logger`.

This module re-exports the runtime logger and adds a small emergency-dump helper
used to guarantee no event metadata is lost when SQLite writes fail.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

# NOTE: Do NOT import src.logging.event_logger at module import time.
# src.logging.event_logger uses this module for emergency dumping, and importing
# it here would create a circular import.
if TYPE_CHECKING:  # pragma: no cover
    from src.logging.event_logger import EventLogger as EventLogger


def __getattr__(name: str):  # pragma: no cover
    """Lazy re-export to keep backwards compatibility without circular imports."""

    if name == "EventLogger":
        from src.logging.event_logger import EventLogger

        return EventLogger
    raise AttributeError(name)


def emergency_dump_event(payload: dict[str, Any], *, path: str = "data/logs/emergency_event_dump.jsonl") -> None:
    """Best-effort JSONL dump for audit survivability.

    Never raises.
    """

    try:
        out = dict(payload)
        out.setdefault("dumped_at", time.time())
        dump_path = Path(path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass
