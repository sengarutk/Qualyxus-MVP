from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger("QualyxusRuntimeMetrics")


class RuntimeMetricsLogger:
    def __init__(self, snapshot_fn, output_path: str = "data/runtime/runtime_metrics.json", interval_seconds: float = 10.0, disk_monitor=None):
        self.snapshot_fn = snapshot_fn
        self.output_path = Path(output_path)
        self.interval_seconds = interval_seconds
        self.disk_monitor = disk_monitor
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(target=self._loop, name="runtime-metrics", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def _get_disk_policy(self):
        if self.disk_monitor:
            return self.disk_monitor.get_policy()
        # Return default policy if no disk monitor
        from src.system.disk_monitor import DiskPolicy
        return DiskPolicy()

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._write_once()
            except Exception as exc:  # never let a transient I/O error kill the metrics thread
                logger.warning("[RuntimeMetrics] Snapshot export failed: %s", exc)
            # Interruptible wait so stop() returns promptly.
            self._stop_event.wait(self.interval_seconds)

    def _memory_mb(self) -> float:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as f:
                rss_pages = int(f.read().split()[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (rss_pages * page_size) / (1024 * 1024)
        except Exception:
            return 0.0

    def _per_camera_memory_mb(self, camera_id: str) -> float:
        # For now, return process memory as we don't have per-thread memory tracking
        # In a production system, this would track memory usage per camera worker
        return self._memory_mb()

    def _write_once(self):
        snapshot = self.snapshot_fn()
        disk_policy = self._get_disk_policy()
        metrics = {
            "timestamp": time.time(),
            "memory_mb": round(self._memory_mb(), 3),
            "disk": {
                "free_percent": round(disk_policy.free_percent, 2),
                "level": disk_policy.level,
                "write_latency_ms": round(disk_policy.write_latency_ms, 2),
                "raw_enabled": disk_policy.raw_enabled,
                "defect_enabled": disk_policy.defect_enabled,
            },
            "cameras": {},
        }
        for cam_id, info in snapshot.items():
            uptime = max(0.001, time.time() - info.get("started_at", time.time()))
            metrics["cameras"][cam_id] = {
                "fps": round(float(info.get("frames_captured", 0)) / uptime, 3),
                "logging_fps": round(float(info.get("events_logged", 0)) / uptime, 3),
                "memory_mb": round(self._per_camera_memory_mb(cam_id), 3),
                "queue_sizes": {
                    "frame": info.get("frame_queue_size", 0),
                    "event": info.get("event_queue_size", 0),
                },
                "dropped_frames": info.get("dropped_frames", 0),
                "backpressure_warnings": info.get("backpressure_warnings", 0),
                "reconnect_count": info.get("reconnect_count", 0),
                "restarts": info.get("restarts", 0),
            }

        # Atomic replace so the dashboard never reads a half-written file.
        tmp_path = self.output_path.with_name(self.output_path.name + ".tmp")
        tmp_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.output_path)
