"""A power cut can leave a database unreadable. That must not stop the inspection line."""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger("SqliteRecovery")


def set_aside_if_corrupt(path: str | Path) -> Path | None:
    """Move a damaged database aside so a fresh one can start. Returns where it went.

    The file is kept for recovery, never deleted. Run this at start-up, before anything opens
    the database. Errors that do not prove damage (permissions, a lock) leave the file alone.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        conn = sqlite3.connect(str(path), timeout=5.0)
        try:
            if conn.execute("PRAGMA quick_check(1)").fetchone()[0] == "ok":
                return None
        finally:
            conn.close()
    except sqlite3.OperationalError as e:
        logger.warning("[Recovery] Could not check %s: %s", path, e)
        return None
    except sqlite3.DatabaseError:
        pass
    destination = path.with_name(f"{path.name}.corrupt-{time.strftime('%Y%m%d-%H%M%S')}")
    os.replace(path, destination)
    for suffix in ("-wal", "-shm"):
        sidecar = Path(f"{path}{suffix}")
        if sidecar.exists():
            os.replace(sidecar, f"{destination}{suffix}")
    logger.critical("[Recovery] %s was damaged. Moved to %s and starting a fresh one.", path, destination)
    return destination
