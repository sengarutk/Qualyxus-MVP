from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT_DIR = _Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_DIR))


import os
import time

from src.camera.camera_manager import CameraManager


def main():
    # Simulate failure conditions:
    # - DB lock (logger retries + graceful failure)
    # - Disk pressure (disable raw/defect saves)
    os.environ["SIMULATE_DB_LOCK"] = "1"
    os.environ["SIMULATED_DISK_FREE_PERCENT"] = "8"

    cfg = {
        "paths": {
            "data_dir": "data",
            "models_dir": "models/checkpoints",
        },
        "logging": {
            "sqlite_path": "data/events/events.db",
            "csv_path": "data/events/events.csv",
        },
        "inference": {
            "conf": 0.35,
            "skip_frames": 1,
        },
        "camera_manager": {
            "monitor_interval": 1.0,
            "restart_backoff": 0.5,
        },
        "watchdog": {
            "heartbeat_timeout": 10.0,
            "frame_drop_threshold": 100,
            "queue_overflow_streak_threshold": 10,
            "base_backoff": 0.5,
            "max_backoff": 4.0,
        },
        "disk_monitor": {
            "watch_path": "data",
            "interval_seconds": 1.0,
            "warning_threshold": 15.0,
            "critical_threshold": 10.0,
        },
        "runtime_metrics": {
            "output_path": "data/runtime/runtime_metrics.json",
            "interval_seconds": 2.0,
        },
        "camera": {
            "fps": 10,
            "width": 320,
            "height": 240,
            "reconnect_delay": 1.0,
            "save_every": 2,
            "frame_queue_size": 4,
            "event_queue_size": 8,
            "raw_dir": "data/live/raw",
            "defect_dir": "data/live/defects",
        },
        "cameras": [
            {
                "id": "cam_drop",
                "source": "synthetic://drop",
                "enabled": True,
                "mock_inference": True,
                "mock_event_every": 1,
                "synthetic_fail_after_frames": 12,
                "independent_logs": True,
            },
            {
                "id": "cam_stable",
                "source": "synthetic://stable",
                "enabled": True,
                "mock_inference": True,
                "mock_event_every": 2,
                "independent_logs": True,
            },
        ],
    }

    manager = CameraManager(config_data=cfg)
    manager.start()

    try:
        time.sleep(8.0)
        snapshot = manager.get_status_snapshot()

        assert "cam_drop" in snapshot and "cam_stable" in snapshot
        assert snapshot["cam_stable"]["frames_captured"] > 0, "stable camera not capturing"
        assert snapshot["cam_drop"]["restarts"] >= 1, "drop camera did not auto-restart"

        # DB lock should not crash process; logger failures are retried then graceful.
        assert snapshot["cam_stable"]["logger_alive"], "logger thread crashed under DB lock"

        # Critical disk simulation should disable both raw and defect writes.
        policy = manager.disk_monitor.get_policy()
        assert policy.level == "critical", "disk critical mode not triggered"
        assert not policy.raw_enabled and not policy.defect_enabled, "disk controls not enforced"

        print("✅ Reliability simulation passed")
    finally:
        manager.stop()
        os.environ.pop("SIMULATE_DB_LOCK", None)
        os.environ.pop("SIMULATED_DISK_FREE_PERCENT", None)


if __name__ == "__main__":
    main()
