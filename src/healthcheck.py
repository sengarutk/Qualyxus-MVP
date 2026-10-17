from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from src.config.config import Config


def _check_db_writable(cfg: Config) -> bool:
    sqlite_path = Path(cfg.get("logging", "sqlite_path", default="data/events/events.db"))
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(str(sqlite_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS __healthcheck (id INTEGER PRIMARY KEY, ts REAL)")
        # Single-row upsert: probes run every 30s forever, so the table must not grow.
        cur.execute("INSERT OR REPLACE INTO __healthcheck (id, ts) VALUES (1, ?)", (time.time(),))
        cur.execute("DELETE FROM __healthcheck WHERE id != 1")
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False


def _check_cameras_active() -> bool:
    status_path = Path(os.getenv("CAMERA_STATUS_PATH", "data/live/camera_status.json"))
    timeout_seconds = float(os.getenv("HEALTH_STATUS_TIMEOUT", "20"))

    if not status_path.exists():
        return False

    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    ts = float(payload.get("timestamp", 0))
    if ts <= 0 or (time.time() - ts) > timeout_seconds:
        return False

    workers = payload.get("workers", {})
    if not workers:
        return False

    # At least one camera should be alive and producing captures
    for _, info in workers.items():
        if info.get("capture_alive") and info.get("frames_captured", 0) > 0:
            return True
    return False


def _camera_ingest_modes() -> list[str]:
    status_path = Path(os.getenv("CAMERA_STATUS_PATH", "data/live/camera_status.json"))
    if not status_path.exists():
        return []
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    workers = payload.get("workers", {}) or {}
    return [str(info.get("ingest_mode", "unknown")) for info in workers.values()]


def main():
    cfg = Config.load()
    mode = os.getenv("HEALTHCHECK_MODE", "inference").lower()

    db_ok = _check_db_writable(cfg)
    if not db_ok:
        print("[QUALYXUS] healthcheck: db not writable", file=sys.stderr)
        raise SystemExit(1)

    if mode == "inference":
        cam_ok = _check_cameras_active()
        if not cam_ok:
            print("[QUALYXUS] healthcheck: cameras not active", file=sys.stderr)
            raise SystemExit(1)
        ingest_modes = _camera_ingest_modes()
        if ingest_modes:
            print(f"[QUALYXUS] healthcheck: ingest_modes={','.join(ingest_modes)}")

    print("[QUALYXUS] healthcheck: ok")


if __name__ == "__main__":
    main()
