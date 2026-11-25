"""Exit 0 only while the station can really inspect: for Docker, systemd and monitoring.

Reads what the runtime itself reports: every camera, the PLC heartbeat, undelivered verdicts
and the disk. A status that is old or missing means the runtime is not working.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from src.config.config import Config


def problems(cfg: Config, now: float | None = None) -> list[str]:
    now = time.time() if now is None else now
    path = Path(os.getenv("CAMERA_STATUS_PATH")
                or cfg.get("camera_manager", "status_path", default=None) or "data/live/camera_status.json")
    max_age = float(os.getenv("HEALTH_STATUS_TIMEOUT", "20"))
    try:
        status = json.loads(path.read_text(encoding="utf-8"))
        age = now - float(status["timestamp"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"no runtime status at {path}: {e}"]
    if age > max_age:
        return [f"the runtime last reported {age:.0f} s ago"]
    found = [str(p) for p in status.get("problems") or []]
    if not status.get("workers"):
        found.append("no cameras are running")
    database = Path(cfg.get("logging", "sqlite_path", default="data/events/events.db"))
    try:
        with sqlite3.connect(f"file:{database}?mode=rw", uri=True, timeout=5.0) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.rollback()
    except sqlite3.Error as e:
        found.append(f"the event database cannot be written: {e}")
    return found


def main() -> int:
    found = problems(Config.load())
    for problem in found:
        print(f"unhealthy: {problem}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
