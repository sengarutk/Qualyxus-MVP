from __future__ import annotations

import json
import time
from pathlib import Path

from src.camera.camera_manager import CameraManager

def main():
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
            "monitor_interval": 0.5,
            "restart_backoff": 0.5,
        },
        "watchdog": {
            "heartbeat_timeout": 2.5,
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
                "id": "cam_alpha",
                "source": "synthetic://alpha",
                "enabled": True,
                "mock_inference": True,
                "mock_event_every": 1,
                "synthetic_fail_after_frames": 10,
                "independent_logs": True,
            },
            {
                "id": "cam_bravo",
                "source": "synthetic://bravo",
                "enabled": True,
                "mock_inference": True,
                "mock_event_every": 1,
                "independent_logs": True,
            },
        ],
    }

    manager = CameraManager(config_data=cfg)
    manager.start()

    snapshots = []
    start = time.time()
    run_seconds = 8.0

    try:
        while time.time() - start < run_seconds:
            time.sleep(1.0)
            snapshots.append(manager.get_status_snapshot())
    finally:
        manager.stop()

    final = manager.get_status_snapshot()
    print("Final status snapshot:")
    print(json.dumps(final, indent=2))

    assert "cam_alpha" in final and "cam_bravo" in final, "Expected two camera workers"
    assert final["cam_bravo"]["frames_captured"] > 0, "cam_bravo did not capture frames"
    assert final["cam_bravo"]["events_logged"] > 0, "cam_bravo did not log events"

    # cam_alpha is configured to fail and should auto-restart at least once
    assert final["cam_alpha"]["restarts"] >= 1, "cam_alpha did not auto-restart"

    csv_dir = Path("data/events")
    alpha_logs = list(csv_dir.glob("events_cam_alpha_*.csv"))
    bravo_logs = list(csv_dir.glob("events_cam_bravo_*.csv"))
    assert alpha_logs, "Independent log file for cam_alpha was not created"
    assert bravo_logs, "Independent log file for cam_bravo was not created"

    print("✅ Isolation test passed")
    print(f"cam_alpha logs: {len(alpha_logs)} | cam_bravo logs: {len(bravo_logs)}")


if __name__ == "__main__":
    main()
