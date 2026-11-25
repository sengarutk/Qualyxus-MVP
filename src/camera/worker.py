"""What one camera's worker holds, and the limits shared by the camera code."""
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from src.logging.schema import SystemState


_TRUTHY = {"1", "true", "yes"}
_SHEDDING_STATES = (SystemState.CRITICAL, SystemState.EMERGENCY)
FRAME_POOL_BYTES = 192 * 1024 * 1024
BACKLOG_BYTES = 1024 * 1024 * 1024
INSPECT_BATCH = 8
EVIDENCE_PENDING = 8
VERDICT_LOST_HOLD_S = 10.0
UNKNOWN_RECIPE = "unknown_recipe"
UNSEEN_CHECK_S = 1.0
UNSEEN_MARGIN_S = 2.0


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in _TRUTHY


class CoercibleAwaitable:
    """Lets synchronous lifecycle methods be used both directly and with `await`."""

    def __init__(self, result=None):
        self.result = result

    def __iter__(self):
        return self

    def __next__(self):
        raise StopIteration(self.result)

    def __await__(self):
        return self


@dataclass
class FramePacket:
    camera_id: str
    session_id: str
    frame_id: int
    timestamp_ms: int
    frame: np.ndarray
    full_image_path: str | None


class FrameSlot:
    def __init__(self, h, w):
        self.frame = np.ascontiguousarray(np.zeros((h, w, 3), dtype=np.uint8))


def pool_slots(height: int, width: int) -> int:
    """Frames a camera keeps: a memory budget, not a count, so 12 MP costs what VGA does."""
    return int(np.clip(FRAME_POOL_BYTES // (height * width * 3), 4, 32))


@dataclass
class CameraWorkerState:
    camera_id: str
    session_id: str
    source: Any
    frame_queue: queue.Queue
    event_queue: queue.Queue
    stop_event: threading.Event

    capture_thread: threading.Thread | None = None
    inference_thread: threading.Thread | None = None

    started_at: float = field(default_factory=time.time)
    last_capture_ts: float = 0.0
    last_infer_ts: float = 0.0
    last_log_ts: float = 0.0
    frames_captured: int = 0
    frames_inferred: int = 0
    events_logged: int = 0
    restarts: int = 0
    reconnect_count: int = 0
    dropped_frames: int = 0
    queue_overflow_streak: int = 0
    last_heartbeat_ts: float = 0.0
    logger_stall_start: float = 0.0
    failed: bool = False
    failure_reason: str | None = None

    events_this_sec: int = 0
    events_sec_start: float = 0.0
    backpressure_warnings: int = 0
    ingest_mode: str = "OpenCV-SW"
    force_mock_inference: bool = False
    last_throttle_log_ts: float = 0.0
    last_geometric_diagnostic: dict = field(default_factory=dict)
    last_geometry_log_ts: float = 0.0
    geometry_inflight: int = 0
    tracker: Any = None
    parts_inspected: int = 0
    parts_failed: int = 0
    parts_not_inspected: int = 0
    not_inspected_logged: bool = False
    pool: list = field(default_factory=list)
    pool_write_index: int = 0
    prev_centroid: tuple[float, float] | None = None

