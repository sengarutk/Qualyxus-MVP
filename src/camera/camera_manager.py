import asyncio
import collections
import copy
import dataclasses
import json
import logging
import os
import queue
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from src.camera.part_inspection import CompletedPart, PartPresenceTracker, PartVerdictFuser, part_pose
from src.config.config import Config
from src.core.factory_manager import FactoryManager
from src.inference.analysis.geometry_engine import GeometryEngine
from src.inference.analysis.vibration_engine import VibrationAnalyzer
from src.inference.postprocess import severity_from_conf
from src.logging.event_logger import EventLogger
from src.logging.schema import DefectEvent, SystemState
from src.system.disk_monitor import DiskMonitor
from src.system.resource_monitor import ResourceGovernor
from src.system.runtime_metrics import RuntimeMetricsLogger
from src.system.watchdog import WatchdogManager
from src.telemetry.plc_adapter import PLC_FLAG_NOT_INSPECTED, PLC_PASS
from src.telemetry.telemetry_provider import TelemetryProvider

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes"}
_SHEDDING_STATES = (SystemState.CRITICAL, SystemState.EMERGENCY)
_POOL_SLOTS = 128


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


def is_part_present(frame: np.ndarray, std_dev_threshold: float = 2.2) -> bool:
    """
    Adaptive multi-angle part presence check.
    Supports low-contrast side profiles (cam_3) via dual std_dev + Laplacian variance checks.
    Returns True if std_dev >= 2.2 OR Laplacian edge variance >= 6.0,
    or if QUALYXUS_SAVE_BLANK_FRAMES=1 override environment flag is enabled.
    """
    if os.getenv("QUALYXUS_SAVE_BLANK_FRAMES", "0").strip().lower() in {"1", "true", "yes"}:
        return True
    if frame is None or frame.size == 0:
        return False
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.shape[2] == 3 else frame[:, :, 0]
    else:
        gray = frame
    # Denoise first: the Laplacian amplifies per-pixel sensor noise, which otherwise makes an
    # empty, noisy background look like a textured part.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    std_dev = float(np.std(gray))
    if std_dev >= std_dev_threshold:
        return True
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return lap_var >= 6.0


class CameraSyncManager:
    """
    Synchronized Multi-Camera Barrier Lock & Sequence Alignment.
    Aligns frame fetches across active channels based on sequence numbers or Presentation Timestamps (PTS).
    Injects non-blocking timeout null-events on dropped frames to keep worker threads in lockstep.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._channels = {}
                    cls._instance._timestamps = {}
                    cls._instance._barrier_lock = threading.Lock()
                    cls._instance._condition = threading.Condition(cls._instance._barrier_lock)
        return cls._instance

    def register_channel(self, camera_id: str):
        with self._barrier_lock:
            self._channels[camera_id] = 0
            self._timestamps[camera_id] = time.perf_counter()

    def unregister_channel(self, camera_id: str):
        with self._barrier_lock:
            self._channels.pop(camera_id, None)
            self._timestamps.pop(camera_id, None)
            self._condition.notify_all()

    def sync_barrier(self, camera_id: str, sequence_id: int, timeout_sec: float = 0.05, pts: float | None = None) -> bool:
        with self._barrier_lock:
            if camera_id not in self._channels:
                return True
            
            curr_pts = pts if pts is not None else time.perf_counter()
            self._channels[camera_id] = sequence_id
            self._timestamps[camera_id] = curr_pts

            # Auto-resync: If any channel falls behind by > 5 steps relative to the fastest channel, advance it
            max_seq = max(self._channels.values()) if self._channels else sequence_id
            for ch, seq in list(self._channels.items()):
                if max_seq - seq > 5:
                    self._channels[ch] = max_seq - 2

            def _is_synced() -> bool:
                if not self._channels:
                    return True
                max_s = max(self._channels.values())
                min_s = min(self._channels.values())
                if (max_s - min_s) <= 5:
                    return True
                max_t = max(self._timestamps.values())
                min_t = min(self._timestamps.values())
                return (max_t - min_t) <= 0.05

            if _is_synced():
                self._condition.notify_all()
                return True

            start_t = time.perf_counter()
            while True:
                max_seq = max(self._channels.values()) if self._channels else sequence_id
                for ch, seq in list(self._channels.items()):
                    if max_seq - seq > 5:
                        self._channels[ch] = max_seq - 2

                if _is_synced():
                    return True

                rem = timeout_sec - (time.perf_counter() - start_t)
                if rem <= 0:
                    return False
                self._condition.wait(timeout=min(0.005, rem))




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
    sync_misses: int = 0
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


class CameraManager:
    """
    High-Velocity Ingestion Pipeline.
    Strictly handles RTSP/USB capture, batched inference hand-off, defect persistence
    and async delegation to the logging, reasoning and reflex layers.
    """
    def __init__(self, config_path: str = "config.yaml", config_data: dict | None = None):
        self.config_path = config_path
        self.raw_config = self._load_raw_config(config_path, config_data)
        self.config = Config(raw=copy.deepcopy(self.raw_config))
        self.main_loop: asyncio.AbstractEventLoop | None = None

        self.shutdown_event = threading.Event()
        self._stopped = False
        self.fast_queue: queue.Queue = queue.Queue(maxsize=20)
        self.workers: dict[str, CameraWorkerState] = {}

        telemetry_cfg = self.raw_config.get("telemetry", {}) or {}
        telemetry_mode = str(telemetry_cfg.get("mode") or "simulated")
        self.telemetry = TelemetryProvider(
            simulation_mode=telemetry_mode == "simulated",
            mode=telemetry_mode,
            protocol=str(telemetry_cfg.get("protocol") or "modbus"),
            endpoint=telemetry_cfg.get("endpoint"),
            port=int(telemetry_cfg.get("port") or 502),
            unit_id=int(telemetry_cfg.get("unit_id") or 1),
            register_map=telemetry_cfg.get("register_map"),
        )

        self.governor = ResourceGovernor(config=self.config)
        FactoryManager().register_instance("ResourceGovernor", self.governor)
        self._last_policy_state: SystemState | None = None
        self._resource_log: collections.deque[str] = collections.deque(maxlen=50)
        # Every capture worker consults the governor once per frame; sample the vitals
        # (psutil/NVML queries) at the configured monitoring interval instead.
        self._policy_cache_ttl = max(0.1, float(self.governor.rm.monitoring_interval))
        self._policy_cache: tuple[float, SystemState] = (0.0, SystemState.SAFE)
        self._policy_lock = threading.Lock()

        manager_cfg = self.raw_config.get("camera_manager", {}) or {}
        self.monitor_interval = float(manager_cfg.get("monitor_interval", 2.0))
        self.restart_backoff = float(manager_cfg.get("restart_backoff", 1.5))
        self.geometry_ok_log_interval = float(manager_cfg.get("geometry_ok_log_interval_sec", 1.0))
        self.status_path = Path(
            os.getenv("CAMERA_STATUS_PATH")
            or manager_cfg.get("status_path")
            or "data/live/camera_status.json"
        )
        data_dir = Path((self.raw_config.get("paths", {}) or {}).get("data_dir") or "data")
        self.system_health_path = data_dir / "runtime" / "system_health.json"

        camera_defaults = self._global_camera_defaults()
        self.raw_dir = Path(camera_defaults.get("raw_dir") or "data/live/raw")
        self.defect_dir = Path(camera_defaults.get("defect_dir") or "data/live/defects")

        hardening_cfg = self.raw_config.get("hardening", {}) or {}
        self.max_pending_events = int(hardening_cfg.get("max_pending_events", 100))
        self.max_events_per_sec = float(hardening_cfg.get("max_events_per_sec", 10.0))

        test_cfg = self.raw_config.get("test_flags", {}) or {}
        self.simulate_camera_drop = bool(test_cfg.get("simulate_camera_drop", False))

        self._lock = threading.Lock()
        self._geometry_lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None
        self._reflex_thread: threading.Thread | None = None

        watchdog_cfg = self.raw_config.get("watchdog", {}) or {}
        self.watchdog = WatchdogManager(
            heartbeat_timeout=float(watchdog_cfg.get("heartbeat_timeout", 10.0)),
            frame_drop_threshold=int(watchdog_cfg.get("frame_drop_threshold", 200)),
            queue_overflow_streak_threshold=int(watchdog_cfg.get("queue_overflow_streak_threshold", 25)),
            base_backoff=float(watchdog_cfg.get("base_backoff", 1.0)),
            max_backoff=float(watchdog_cfg.get("max_backoff", 30.0)),
            logger_stall_timeout=float(hardening_cfg.get("logger_stall_timeout", 5.0)),
            frame_drop_window_seconds=float(watchdog_cfg.get("frame_drop_window_seconds", 60.0)),
        )

        disk_cfg = self.raw_config.get("disk_monitor", {}) or {}
        self.disk_monitor = DiskMonitor(
            watch_path=disk_cfg.get("watch_path", "data"),
            interval_seconds=float(disk_cfg.get("interval_seconds", 60.0)),
            warning_threshold=float(disk_cfg.get("warning_threshold", 15.0)),
            critical_threshold=float(disk_cfg.get("critical_threshold", 10.0)),
            raw_frames_dir=str(self.raw_dir),
        )

        metrics_cfg = self.raw_config.get("runtime_metrics", {}) or {}
        self.runtime_metrics = RuntimeMetricsLogger(
            snapshot_fn=self.get_status_snapshot,
            output_path=str(metrics_cfg.get("output_path") or "data/runtime/runtime_metrics.json"),
            interval_seconds=float(metrics_cfg.get("interval_seconds", 10.0)),
            disk_monitor=self.disk_monitor,
        )

        self.vibration_analyzers: dict[str, VibrationAnalyzer] = {}
        self.geometry_engines: dict[str, GeometryEngine] = {}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="geometry-worker")
        self.vlm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-reasoner")
        # Advisory flags keep their order and never wait behind slow VLM narration.
        self.plc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="plc-writer")
        vibration_cfg = self.raw_config.get("vibration_analysis", {}) or {}
        self._kap_db_path: str = str(vibration_cfg.get("kap_db_path", "data/knowledge/knowledge_base.db"))
        # Optical "vibration" is centroid motion of detections sampled at the camera frame
        # rate: it only resolves < fps/2 Hz and is not a mm/s velocity unless calibrated.
        # Off by default; a real accelerometer feed is the proper vibration source.
        self.vibration_enabled = bool(vibration_cfg.get("enabled", False))
        scale = vibration_cfg.get("pixel_to_mm_scale")
        self.vibration_scale = float(scale) if scale else None

        vlm_cfg = self.raw_config.get("vlm_reasoning", {}) or {}
        self.vlm_enabled = bool(vlm_cfg.get("enabled", True))
        self.vlm_max_pending = max(1, int(vlm_cfg.get("max_pending", 8)))
        self._vlm_pending = 0
        self._vlm_lock = threading.Lock()

        self.disk_write_queue: queue.Queue = queue.Queue(maxsize=128)
        self._disk_writer_thread: threading.Thread | None = None
        self._last_missing_logger_warning = 0.0

        from src.inference.multiplexer import InferenceMultiplexer
        self.multiplexer = InferenceMultiplexer(config=self.config)
        FactoryManager().register_instance("InferenceMultiplexer", self.multiplexer)

        inference_cfg = self.raw_config.get("inference", {}) or {}
        # How long a capture worker waits for its batched inference result. GPU batches return
        # well inside one frame interval; CPU inference needs a much longer budget or every
        # result would be discarded before it arrives.
        default_timeout = 0.033 if self.multiplexer.device == "cuda" else 0.5
        self.result_timeout_sec = float(inference_cfg.get("result_timeout_sec") or default_timeout)
        self.defect_class_ids = self._resolve_defect_classes(inference_cfg.get("defect_classes"))

        # Per-part inspection: one pass/fail record per physical part (labels for the
        # process model), judged by the good-parts-only anomaly detector.
        inspection_cfg = self.raw_config.get("inspection", {}) or {}
        self.inspection_enabled = bool(inspection_cfg.get("enabled", True))
        self.presence_std_threshold = float(inspection_cfg.get("presence_std_threshold", 2.2))
        self._presence_on_frames = int(inspection_cfg.get("on_frames", 2))
        self._presence_off_frames = int(inspection_cfg.get("off_frames", 2))
        self.part_trackers: dict[str, PartPresenceTracker] = {}
        self.inspection_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="part-inspection")
        self.part_fuser = PartVerdictFuser(
            (str(c.get("id")) for c in (self.raw_config.get("cameras") or [])
             if c.get("enabled", True) and c.get("required_view")),
            timeout_s=float((self.raw_config.get("inspection") or {}).get("view_timeout_s", 5.0)),
        )
        self.heartbeat_s = float((self.raw_config.get("telemetry") or {}).get("heartbeat_s", 1.0))
        self._last_heartbeat = 0.0
        self._part_counter = 0
        self._inspection_faulted = False
        # Models per recipe (registry-versioned with shadow + hot reload, or explicit config paths).
        from src.inference.inspection_stack import StackProvider

        self.stack_provider = StackProvider(inspection_cfg)
        self.camera_recipes = {str(c.get("id")): c.get("recipe") for c in (self.raw_config.get("cameras") or [])
                               if c.get("id") is not None and c.get("recipe")}
        self.recipe_from_scalar = inspection_cfg.get("recipe_from_scalar")
        self.stack_provider.get()  # load the default recipe's models at start-up, not on the first part
        self._health_cfg = inspection_cfg.get("health", {}) or {}
        self.model_health: dict[Any, Any] = {}
        self.retrain_on_drift = bool(self._health_cfg.get("request_retrain", True))
        review_cfg = inspection_cfg.get("review", {}) or {}
        self.review_queue = None
        if review_cfg.get("enabled", True):
            from src.inference.anomaly.monitoring import ReviewQueue

            self.review_queue = ReviewQueue(review_cfg.get("dir", "data/review"),
                                            band=tuple(review_cfg.get("band", (0.8, 1.25))))
        audit_cfg = inspection_cfg.get("audit", {}) or {}
        self.audit_log = None
        if audit_cfg.get("enabled", True):
            from src.inference.audit_log import InspectionAuditLog

            self.audit_log = InspectionAuditLog(audit_cfg.get("dir", "data/audit"), audit_cfg.get("key_file"))
        self._pool_cfg = inspection_cfg.get("good_pool", {}) or {}
        self._good_pools: dict[str, Any] = {}
        self.process_service = None
        self._part_feeds: dict[str, Any] = {}

        if not FactoryManager().get_instance("EventLogger"):
            logging_cfg = self.raw_config.get("logging", {}) or {}
            base_sqlite = logging_cfg.get("sqlite_path", "data/events/events.db")
            base_csv = logging_cfg.get("csv_path", "data/events/events.csv")
            event_logger = EventLogger(sqlite_path=base_sqlite, csv_path=base_csv)
            FactoryManager().register_instance("EventLogger", event_logger)
        FactoryManager().register_instance("CameraManager", self)

    def _resolve_defect_classes(self, configured) -> set[int] | None:
        """Which frame-detector classes are defects. None = every class; empty = none.

        A generic COCO model detects everyday objects, not defects: unless defect classes
        are configured explicitly, its detections must never be logged as defects.
        """
        names = getattr(self.multiplexer, "class_names", {}) or {}
        if configured:
            wanted = {str(c) for c in configured}
            ids = {i for i, n in names.items() if n in wanted or str(i) in wanted}
            missing = wanted - {names.get(i, str(i)) for i in ids} - {str(i) for i in ids}
            if missing:
                logger.warning("[Inference] Configured defect classes not in the model: %s", sorted(missing))
            return ids
        if "person" in names.values() and len(names) >= 80:
            logger.warning(
                "[Inference] %s is a generic COCO detector; frame-level defect events are disabled. "
                "Train a defect model or set inference.defect_classes. Parts are still judged by the "
                "good-parts-only anomaly detector.", self.config.get("inference", "model_path"),
            )
            return set()
        return None

    def _filter_defect_detections(self, boxes: np.ndarray, xyxy: np.ndarray, conf: np.ndarray):
        if self.defect_class_ids is None or len(boxes) == 0:
            return boxes, xyxy, conf
        if boxes.ndim < 2 or boxes.shape[1] <= 5:
            return boxes, xyxy, conf  # no class column (synthetic/mock boxes)
        keep = np.isin(boxes[:, 5].astype(int), list(self.defect_class_ids))
        return boxes[keep], xyxy[keep], conf[keep]

    # Default-recipe models (status display, backwards compatibility).
    @property
    def anomaly_detector(self):
        return self.stack_provider.get()[0].anomaly

    @property
    def segmenter(self):
        return self.stack_provider.get()[0].segmenter

    @property
    def type_classifier(self):
        return self.stack_provider.get()[0].typer

    def set_recipe(self, camera_id: str, recipe: str | None) -> None:
        """Switch the product/recipe a camera inspects (models load on its next part)."""
        if recipe:
            self.camera_recipes[camera_id] = recipe
        else:
            self.camera_recipes.pop(camera_id, None)
        logger.info("[Inspection] %s now inspects recipe '%s'", camera_id, recipe or self.stack_provider.default_recipe)

    def _recipe_for(self, camera_id: str, cycle_id: str | None) -> str:
        """Recipe from the producing cycle's machine data (e.g. mold id), else the camera's, else default."""
        if self.recipe_from_scalar and cycle_id and self.process_service is not None:
            try:
                cycle = self.process_service.store.get_cycle(cycle_id)
                value = (cycle.scalars or {}).get(self.recipe_from_scalar) if cycle else None
                if value:
                    return str(value)
            except Exception as e:
                logger.debug("[Inspection] recipe lookup for %s failed: %s", cycle_id, e)
        return self.camera_recipes.get(camera_id) or self.stack_provider.default_recipe

    def _good_pool(self, recipe: str):
        if not self._pool_cfg.get("enabled", True):
            return None
        pool = self._good_pools.get(recipe)
        if pool is None:
            from src.inference.good_pool import GoodPartPool

            pool = self._good_pools.setdefault(recipe, GoodPartPool(
                self._pool_cfg.get("dir", "data/good_pool"), recipe,
                every_n=int(self._pool_cfg.get("every_n", 5)), max_items=int(self._pool_cfg.get("max_items", 3000)),
                max_score=self._pool_cfg.get("max_score", 0.8)))
        return pool

    def attach_process_service(self, service) -> None:
        """Link inspections to machine cycles; render simulated parts for synthetic scenes."""
        self.process_service = service
        source = getattr(service, "source", None)
        if source is None or not hasattr(source, "outcomes_ended_between"):
            return
        from src.process.synthetic_parts import SyntheticPartFeed

        for cam in self.raw_config.get("cameras", []) or []:
            if str(cam.get("synthetic_scene", "")).lower() != "process":
                continue
            link = service.aligner.link_for(str(cam.get("id")))
            delay = link.transport_delay_s if link else 4.0
            self._part_feeds[str(cam.get("id"))] = SyntheticPartFeed(
                source, transport_delay_s=delay, dwell_s=float(cam.get("part_dwell_s", 0.6)),
                view_index=int(cam.get("view_index", 0)), n_views=int(cam.get("n_views", 1)),
                inject_defect_every=int(cam.get("inject_defect_every", 0)),
            )

    def _telemetry_context(self, camera_id: str, ts: float | None = None, cycle_id: str | None = None):
        """Telemetry of the cycle that produced the part in view (falls back to live values)."""
        ctx = self.telemetry.get_telemetry(camera_id)
        service = self.process_service
        if service is None:
            return ctx
        try:
            if cycle_id is None:
                cycle_id = service.aligner.align(camera_id, ts if ts is not None else time.time()).cycle_id
            overrides = {k: v for k, v in service.telemetry_for(cycle_id).items() if v is not None}
            known = {f.name for f in dataclasses.fields(ctx)}
            return dataclasses.replace(ctx, **{k: v for k, v in overrides.items() if k in known})
        except Exception as e:
            logger.debug("[%s] Cycle telemetry unavailable: %s", camera_id, e)
            return ctx

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def enqueue_disk_write(self, target_path: Path | str, image: np.ndarray) -> bool:
        """Offloads cv2.imwrite calls to a background queue. Returns False when the write was shed."""
        try:
            self.disk_write_queue.put_nowait((Path(target_path), image.copy()))
            return True
        except queue.Full:
            return False
        except Exception as e:
            logger.error("[DiskWriter] Failed to enqueue %s: %s", target_path, e)
            return False

    def _disk_writer_loop(self):
        """Dedicated background thread for non-blocking disk persistence."""
        while not self.shutdown_event.is_set() or not self.disk_write_queue.empty():
            try:
                target_path, image = self.disk_write_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(target_path), image):
                    logger.error("[DiskWriter] cv2.imwrite returned False for %s", target_path)
            except Exception as e:
                logger.error("[DiskWriter] Non-blocking imwrite failed for %s: %s", target_path, e)

    def _maybe_save_raw_frame(self, camera_id: str, frame: np.ndarray, frame_id: int, save_every: int) -> None:
        """Persist every Nth non-blank raw frame while the disk policy allows raw saves."""
        if save_every <= 0 or frame_id % save_every != 0:
            return
        try:
            if self.disk_monitor.allow_raw_save() and is_part_present(frame):
                self.enqueue_disk_write(self.raw_dir / f"{camera_id}_frame_{frame_id:04d}.png", frame)
        except Exception as disk_err:
            logger.error("[%s] Failed to persist raw frame: %s", camera_id, disk_err)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_raw_config(config_path: str, config_data: dict | None) -> dict:
        if config_data is not None:
            return copy.deepcopy(config_data)
        return Config.load(config_path=config_path).raw

    def _global_camera_defaults(self) -> dict:
        return self.raw_config.get("camera", {}) or {}

    def _camera_setting(self, camera_cfg: dict, key: str, fallback: Any) -> Any:
        """Per-camera value, falling back to the global `camera:` defaults, then `fallback`."""
        if camera_cfg.get(key) is not None:
            return camera_cfg[key]
        value = self._global_camera_defaults().get(key)
        return fallback if value is None else value

    def _camera_entries(self) -> list[dict]:
        cameras = self.raw_config.get("cameras", [])
        if not cameras:
            raise ValueError("No cameras configured.")
        return [c for c in cameras if c.get("enabled", True)]

    def _make_session_id(self, camera_id: str) -> str:
        return f"{camera_id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _start_camera_worker(self, camera_cfg: dict, previous_restarts: int = 0):
        camera_id = str(camera_cfg.get("id", "camera"))

        use_synth_env = _env_flag("QUALYXUS_USE_SYNTHETIC_CAMERAS")
        force_mock_env = _env_flag("QUALYXUS_FORCE_MOCK_INFERENCE")

        camera_cfg = dict(camera_cfg or {})
        if use_synth_env:
            camera_cfg["source"] = f"synthetic://{camera_id}"
            camera_cfg["mock_inference"] = True
            camera_cfg.setdefault("fallback_to_synthetic_on_failure", True)
        if force_mock_env:
            camera_cfg["mock_inference"] = True

        source = camera_cfg.get("source", 0)
        session_id = self._make_session_id(camera_id)

        state = CameraWorkerState(
            camera_id=camera_id,
            session_id=session_id,
            source=source,
            frame_queue=queue.Queue(maxsize=max(1, int(self._camera_setting(camera_cfg, "frame_queue_size", 16)))),
            event_queue=queue.Queue(maxsize=max(1, int(self._camera_setting(camera_cfg, "event_queue_size", 64)))),
            stop_event=threading.Event(),
            restarts=previous_restarts,
        )

        state.capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{camera_id}",
            args=(state, camera_cfg),
            daemon=True,
        )

        if bool(camera_cfg.get("mock_inference", False)):
            state.inference_thread = threading.Thread(
                target=self._mock_inference_loop,
                name=f"infer-mock-{camera_id}",
                args=(state, camera_cfg),
                daemon=True,
            )

        from src.inference.analysis.tracker import SpatialTemporalTracker
        state.tracker = SpatialTemporalTracker()

        with self._lock:
            self.workers[camera_id] = state

        # Analyzers survive worker restarts: re-creating them would discard the vibration
        # history and reload the golden template on every watchdog respawn.
        if self.vibration_enabled and camera_id not in self.vibration_analyzers:
            self.vibration_analyzers[camera_id] = VibrationAnalyzer(
                buffer_size=512,
                kap_db_path=self._kap_db_path,
                fast_queue=self.fast_queue,
                camera_id=camera_id,
                pixel_to_mm_scale=self.vibration_scale or 1.0,
                calibrated=self.vibration_scale is not None,
            )

        if camera_id not in self.geometry_engines:
            cam_template = f"data/templates/golden_part_{camera_id}.png"
            generic_template = "data/templates/golden_part.png"
            if os.path.exists(cam_template):
                template_path = cam_template
            elif os.path.exists(generic_template):
                template_path = generic_template
            else:
                template_path = None
            self.geometry_engines[camera_id] = GeometryEngine(camera_id=camera_id, template_path=template_path)

        state.capture_thread.start()
        if state.inference_thread:
            state.inference_thread.start()

        logger.info("[CameraManager] Started worker: %s", camera_id)

    def start(self, loop=None) -> CoercibleAwaitable:
        try:
            self.main_loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            self.main_loop = None
        self.shutdown_event.clear()
        self._stopped = False
        self.disk_monitor.start()

        self._disk_writer_thread = threading.Thread(target=self._disk_writer_loop, name="camera-disk-writer", daemon=True)
        self._disk_writer_thread.start()

        self._reflex_thread = threading.Thread(target=self._reflex_worker, name="camera-reflex", daemon=True)
        self._reflex_thread.start()

        try:
            if _psutil is not None:
                proc = _psutil.Process()
                proc.nice(_psutil.HIGH_PRIORITY_CLASS if os.name == "nt" else -15)
        except Exception:
            pass

        for cam in self._camera_entries():
            self._start_camera_worker(cam)

        self._monitor_thread = threading.Thread(target=self._monitor_loop, name="camera-monitor", daemon=True)
        self._monitor_thread.start()
        self.runtime_metrics.start()
        logger.info("[CameraManager] Ingestion pipeline started")
        return CoercibleAwaitable(None)

    def stop_all_streams(self) -> None:
        """Synchronous termination of all active camera ingestion threads."""
        self._sync_stop()

    def _sync_stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.shutdown_event.set()

        with self._lock:
            states = list(self.workers.values())

        # 1) Stop the producers first so no new frames enter the pipeline.
        for state in states:
            state.stop_event.set()
        for state in states:
            for t in (state.capture_thread, state.inference_thread):
                if t is not None and t.is_alive():
                    t.join(timeout=5.0)

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=self.monitor_interval + 2.0)

        # 2) Then stop the shared inference daemon once nothing is waiting on it.
        if hasattr(self, "multiplexer"):
            self.multiplexer.stop()

        # 3) Finally drain the background writers / reflex path.
        for t in (self._reflex_thread, self._disk_writer_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)

        self.executor.shutdown(wait=True, cancel_futures=True)
        self.inspection_executor.shutdown(wait=True, cancel_futures=True)
        self.vlm_executor.shutdown(wait=True, cancel_futures=True)
        self.plc_executor.shutdown(wait=True)
        self.runtime_metrics.stop()
        self.disk_monitor.stop()

    def stop(self) -> CoercibleAwaitable:
        self.stop_all_streams()
        logger.info("[CameraManager] Ingestion pipeline stopped")
        return CoercibleAwaitable(None)

    def get_status_snapshot(self) -> dict[str, dict[str, Any]]:
        """Fetch active processing metrics and lifecycle status across all camera streams."""
        event_logger = FactoryManager().get_instance("EventLogger")
        logger_alive = False
        if event_logger:
            worker_task = getattr(event_logger, "worker_task", None)
            if worker_task is not None:
                logger_alive = not worker_task.done()
            else:
                logger_alive = bool(getattr(event_logger, "_initialized", False))

        with self._lock:
            out = {}
            for camera_id, state in self.workers.items():
                capture_alive = state.capture_thread is not None and state.capture_thread.is_alive()
                out[camera_id] = {
                    "session_id": state.session_id,
                    "ingest_mode": state.ingest_mode,
                    "failed": state.failed,
                    "failure_reason": state.failure_reason,
                    "frames_captured": state.frames_captured,
                    "frames_inferred": state.frames_inferred,
                    "events_logged": state.events_logged,
                    "frame_queue_size": state.frame_queue.qsize(),
                    "event_queue_size": state.event_queue.qsize(),
                    "last_capture_ts": state.last_capture_ts,
                    "last_infer_ts": state.last_infer_ts,
                    "last_log_ts": state.last_log_ts,
                    "restarts": state.restarts,
                    "reconnect_count": state.reconnect_count,
                    "dropped_frames": state.dropped_frames,
                    "sync_misses": state.sync_misses,
                    "backpressure_warnings": state.backpressure_warnings,
                    "queue_overflow_streak": state.queue_overflow_streak,
                    "last_heartbeat_ts": state.last_heartbeat_ts,
                    "started_at": state.started_at,
                    "capture_alive": capture_alive,
                    "inference_alive": bool(state.inference_thread and state.inference_thread.is_alive()),
                    "logger_alive": logger_alive,
                    "last_geometric_diagnostic": dict(state.last_geometric_diagnostic or {}),
                    "parts_inspected": state.parts_inspected,
                    "parts_failed": state.parts_failed,
                    "parts_not_inspected": state.parts_not_inspected,
                    "recipe": self.camera_recipes.get(camera_id) or self.stack_provider.default_recipe,
                    "model_health": dict(getattr(self, "_health_status", {}).get(camera_id, {})),
                }
            return out

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def _open_camera(self, source: Any, width: int = 640, height: int = 480, fps: float = 30.0):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            return (cap, "OpenCV-HW" if str(source).startswith("rtsp://") else "OpenCV-SW")
        cap.release()
        return (None, "OpenCV-SW")

    def _interruptible_sleep(self, seconds: float, state: CameraWorkerState) -> bool:
        """Interruptible zero-CPU sleep that prevents framework pacing distortions."""
        end_time = time.perf_counter() + seconds
        while time.perf_counter() < end_time:
            if self.shutdown_event.is_set() or state.stop_event.is_set():
                return False
            remaining = end_time - time.perf_counter()
            if remaining <= 0:
                break
            state.stop_event.wait(min(0.1, remaining))
        return not (self.shutdown_event.is_set() or state.stop_event.is_set())

    def _current_policy_state(self) -> SystemState:
        with self._policy_lock:
            sampled_at, policy = self._policy_cache
            now = time.monotonic()
            if now - sampled_at >= self._policy_cache_ttl:
                try:
                    policy = self.governor.get_policy_state()
                except Exception as e:
                    logger.debug("[CameraManager] Governor evaluation failed: %s", e)
                self._policy_cache = (now, policy)
            return policy

    def _should_shed_load(self, state: CameraWorkerState) -> bool:
        """True while the resource governor reports CRITICAL/EMERGENCY pressure."""
        if not self.governor:
            return False
        policy = self._current_policy_state()
        if policy not in _SHEDDING_STATES:
            return False
        current_ts = time.time()
        if current_ts - state.last_throttle_log_ts > 1.0:
            logger.warning("[%s] Governor %s: resources saturated. Aggressive shedding active.", state.camera_id, policy.value)
            state.last_throttle_log_ts = current_ts
        return True

    def _capture_loop(self, state: CameraWorkerState, camera_cfg: dict):
        width = int(self._camera_setting(camera_cfg, "width", 640))
        height = int(self._camera_setting(camera_cfg, "height", 480))
        fps = float(self._camera_setting(camera_cfg, "fps", 20))
        reconnect_delay = float(self._camera_setting(camera_cfg, "reconnect_delay", 5.0))
        save_every = int(self._camera_setting(camera_cfg, "save_every", 10))

        is_synthetic = isinstance(state.source, str) and state.source.startswith("synthetic://")
        mock_inference_cfg = bool(camera_cfg.get("mock_inference", False))

        # Register channel with multi-camera sync barrier
        sync_mgr = CameraSyncManager()
        sync_mgr.register_channel(state.camera_id)

        # Pre-allocate the mutable ring buffer managed within CameraWorkerState
        state.pool = [FrameSlot(height, width) for _ in range(_POOL_SLOTS)]
        state.pool_write_index = 0

        cap = None
        if not is_synthetic:
            cap, state.ingest_mode = self._open_camera(state.source, width=width, height=height, fps=fps)
        else:
            state.ingest_mode = "Synthetic"

        frame_interval = 1.0 / max(1.0, fps)
        frame_id = 0

        try:
            while not self.shutdown_event.is_set() and not state.stop_event.is_set():

                if self._should_shed_load(state):
                    if cap is not None and not is_synthetic:
                        cap.grab()  # keep the RTSP buffer drained while shedding
                    state.last_heartbeat_ts = time.perf_counter()
                    self._interruptible_sleep(0.1, state)
                    continue

                loop_start = time.perf_counter()

                if is_synthetic:
                    synthetic_fail_after = camera_cfg.get("synthetic_fail_after_frames")
                    if synthetic_fail_after is not None and state.restarts == 0 and frame_id >= int(synthetic_fail_after):
                        raise RuntimeError("Synthetic capture failure injected")
                    slot = state.pool[state.pool_write_index % _POOL_SLOTS]
                    part_feed = self._part_feeds.get(state.camera_id)
                    if part_feed is not None:
                        rendered, _ = part_feed.render(time.time(), width, height)
                        np.copyto(slot.frame, rendered)
                        frame = slot.frame
                        state.pool_write_index += 1
                        if not sync_mgr.sync_barrier(state.camera_id, frame_id, timeout_sec=0.05, pts=time.perf_counter()):
                            state.sync_misses += 1
                        frame_id += 1
                        state.frames_captured = frame_id
                        state.last_capture_ts = time.perf_counter()
                        state.last_heartbeat_ts = state.last_capture_ts
                        self._dispatch_frame(state, camera_cfg, frame, frame_id, save_every, is_synthetic=True,
                                             mock=mock_inference_cfg or state.force_mock_inference)
                        elapsed = time.perf_counter() - loop_start
                        if elapsed < frame_interval and not self._interruptible_sleep(frame_interval - elapsed, state):
                            break
                        continue
                    slot.frame[:] = (40, 40, 40)

                    # Render dynamic synthetic part geometry with high-contrast borders and text overlay
                    cx = int((320 + (frame_id * 8)) % width)
                    cy = int((240 + (frame_id * 5)) % height)
                    top_left = (max(0, cx - 60), max(0, cy - 40))
                    bottom_right = (min(width - 1, cx + 60), min(height - 1, cy + 40))
                    cv2.rectangle(slot.frame, top_left, bottom_right, (180, 180, 180), -1)
                    cv2.rectangle(slot.frame, top_left, bottom_right, (255, 255, 255), 2)
                    cv2.putText(slot.frame, f"{state.camera_id} - FRAME {frame_id:04d}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                    frame = slot.frame
                    state.pool_write_index += 1
                    if not sync_mgr.sync_barrier(state.camera_id, frame_id, timeout_sec=0.05, pts=time.perf_counter()):
                        state.sync_misses += 1
                else:
                    if cap is None:
                        fallback_enabled = bool(camera_cfg.get("fallback_to_synthetic_on_failure", False)) or _env_flag("QUALYXUS_USE_SYNTHETIC_CAMERAS")
                        if fallback_enabled:
                            logger.warning("[%s] Stream open failed; transitioning immediately to synthetic capture mode.", state.camera_id)
                            state.source = f"synthetic://{state.camera_id}"
                            is_synthetic = True
                            state.ingest_mode = "Synthetic"
                            sync_mgr.register_channel(state.camera_id)
                            continue

                        sync_mgr.unregister_channel(state.camera_id)
                        if not self._interruptible_sleep(reconnect_delay, state):
                            break
                        cap, state.ingest_mode = self._open_camera(state.source, width=width, height=height, fps=fps)
                        if cap is None:
                            continue
                        sync_mgr.register_channel(state.camera_id)
                        state.reconnect_count += 1

                    if str(state.source).startswith("rtsp://") and state.last_capture_ts > 0:
                        elapsed_since_last = time.perf_counter() - state.last_capture_ts
                        if elapsed_since_last > (frame_interval * 1.5):
                            # Skip stale buffered frames so inference sees the live scene.
                            for _ in range(2):
                                cap.grab()

                    grabbed = cap.grab()
                    if not sync_mgr.sync_barrier(state.camera_id, frame_id, timeout_sec=0.05, pts=time.perf_counter()):
                        state.sync_misses += 1

                    frame = None
                    if grabbed:
                        slot = state.pool[state.pool_write_index % _POOL_SLOTS]
                        ret, decoded_img = cap.retrieve(image=slot.frame)
                        if ret and decoded_img is not None:
                            if decoded_img is not slot.frame:
                                # The stream resolution differs from the configured one; re-size the
                                # ring buffer once so subsequent decodes land in pre-allocated memory.
                                h, w = decoded_img.shape[:2]
                                logger.warning(
                                    "[%s] Stream delivers %dx%d (configured %dx%d). Re-allocating frame pool.",
                                    state.camera_id, w, h, width, height,
                                )
                                width, height = w, h
                                state.pool = [FrameSlot(h, w) for _ in range(_POOL_SLOTS)]
                                state.pool_write_index = 0
                                slot = state.pool[0]
                                np.copyto(slot.frame, decoded_img)
                            frame = slot.frame
                            state.pool_write_index += 1

                    if frame is None:
                        cap.release()
                        cap = None
                        sync_mgr.unregister_channel(state.camera_id)
                        if not self._interruptible_sleep(reconnect_delay, state):
                            break
                        continue

                frame_id += 1
                state.frames_captured = frame_id
                state.last_capture_ts = time.perf_counter()
                state.last_heartbeat_ts = state.last_capture_ts

                self._dispatch_frame(state, camera_cfg, frame, frame_id, save_every, is_synthetic=is_synthetic,
                                     mock=mock_inference_cfg or state.force_mock_inference)

                elapsed = time.perf_counter() - loop_start
                if elapsed < frame_interval:
                    if not self._interruptible_sleep(frame_interval - elapsed, state):
                        break
        except BaseException as e:
            state.failed = True
            state.failure_reason = f"{type(e).__name__}: {str(e)}"
            logger.error("[%s] Capture loop exception: %s", state.camera_id, state.failure_reason, exc_info=True)
            state.stop_event.set()
        finally:
            sync_mgr.unregister_channel(state.camera_id)
            logger.info("[%s] Capture loop exited. shutdown=%s, stop=%s", state.camera_id, self.shutdown_event.is_set(), state.stop_event.is_set())
            if cap is not None:
                cap.release()

    def _dispatch_frame(self, state: CameraWorkerState, camera_cfg: dict, frame: np.ndarray, frame_id: int,
                        save_every: int, *, is_synthetic: bool, mock: bool) -> None:
        """Route a captured frame to live batched inference, or to the mock-inference worker."""
        if not mock:
            self._maybe_save_raw_frame(state.camera_id, frame, frame_id, save_every)
            new_events: tuple[list[str], list[str]] = ([], [])
            try:
                multiplexer = FactoryManager().get_instance("InferenceMultiplexer")
                result = None
                if multiplexer:
                    try:
                        request_id = multiplexer.submit_frame(state.camera_id, frame, block=False, timeout=None)
                        result = multiplexer.retrieve_result(request_id, timeout_sec=self.result_timeout_sec)
                    except queue.Full:
                        state.backpressure_warnings += 1
                        result = None

                if result is None:
                    if not is_synthetic:
                        state.dropped_frames += 1
                else:
                    boxes, _masks, xyxy, conf = result
                    boxes, xyxy, conf = self._filter_defect_detections(boxes, xyxy, conf)
                    new_events = self._handle_inference_result(
                        state, frame, frame_id, boxes, xyxy, conf,
                        defect_type="anomaly",
                        severity_fn=severity_from_conf,
                    )
            except Exception as e:
                logger.error("[%s] Multiplexer handoff failed: %s", state.camera_id, e, exc_info=True)
            self._track_part(state, frame, frame_id, *new_events)
            return

        packet = FramePacket(
            camera_id=state.camera_id,
            session_id=state.session_id,
            frame_id=frame_id,
            timestamp_ms=int(time.time() * 1000),
            frame=frame.copy(),
            full_image_path="",
        )
        try:
            state.frame_queue.put_nowait(packet)
            state.queue_overflow_streak = 0
        except queue.Full:
            state.queue_overflow_streak += 1
            if not is_synthetic:
                state.dropped_frames += 1

    def _mock_inference_loop(self, state: CameraWorkerState, camera_cfg: dict):
        # Process-driven synthetic scenes render real defects; injected mock boxes would
        # contaminate the labels, so they are only used for plain synthetic streams.
        process_scene = state.camera_id in self._part_feeds
        mock_event_every = max(1, int(camera_cfg.get("mock_event_every", 10) or 10))
        save_every = int(self._camera_setting(camera_cfg, "save_every", 10))
        manager = FactoryManager()

        try:
            while not (self.shutdown_event.is_set() or state.stop_event.is_set()):
                try:
                    packet: FramePacket = state.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                self._maybe_save_raw_frame(state.camera_id, packet.frame, packet.frame_id, save_every)

                multiplexer = manager.get_instance("InferenceMultiplexer")
                if not multiplexer:
                    self._interruptible_sleep(0.01, state)
                    continue

                try:
                    request_id = multiplexer.submit_frame(state.camera_id, packet.frame)
                    result = multiplexer.retrieve_result(request_id, timeout_sec=1.0)
                except queue.Full:
                    state.backpressure_warnings += 1
                    result = None

                if result is None:
                    is_synthetic = isinstance(state.source, str) and state.source.startswith("synthetic://")
                    if not is_synthetic:
                        state.dropped_frames += 1
                    self._track_part(state, packet.frame, packet.frame_id, [], [])
                    self._interruptible_sleep(0.01, state)
                    continue

                boxes, _masks, xyxy, conf = result
                boxes, xyxy, conf = self._filter_defect_detections(boxes, xyxy, conf)
                if not process_scene and len(boxes) == 0 and packet.frame_id % mock_event_every == 0:
                    offset = (packet.frame_id % 10) * 100.0
                    x1, y1, x2, y2 = 10.0 + offset, 10.0, 90.0 + offset, 90.0
                    boxes = np.array([[x1, y1, x2, y2, 0.99, 0.0]], dtype=np.float32)
                    xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                    conf = np.array([0.99], dtype=np.float32)

                new_events = self._handle_inference_result(
                    state, packet.frame, packet.frame_id, boxes, xyxy, conf,
                    defect_type="mock_defect",
                    severity_fn=lambda _c: 3,
                )
                self._track_part(state, packet.frame, packet.frame_id, *new_events)
        except Exception as e:
            state.failed = True
            state.failure_reason = f"{type(e).__name__}: {str(e)}"
            logger.error("[%s] Mock inference loop exception: %s", state.camera_id, state.failure_reason, exc_info=True)
            state.stop_event.set()

    # ------------------------------------------------------------------
    # Shared post-inference handling (live + mock paths)
    # ------------------------------------------------------------------

    def _handle_inference_result(
        self,
        state: CameraWorkerState,
        frame: np.ndarray,
        frame_id: int,
        boxes: np.ndarray,
        xyxy: np.ndarray,
        conf: np.ndarray,
        *,
        defect_type: str,
        severity_fn,
    ) -> tuple[list[str], list[str]]:
        """Track detections, log new defects; returns (new event ids, their defect types)."""
        state.frames_inferred += 1
        state.last_infer_ts = time.perf_counter()

        num_boxes = len(boxes)
        detections = np.empty((num_boxes, 6), dtype=np.float32)
        if num_boxes > 0:
            detections[:, :4] = xyxy
            detections[:, 4] = conf
            detections[:, 5] = boxes[:, 5] if boxes.shape[1] > 5 else 0.0

        tracks = state.tracker.update(detections) if state.tracker is not None else []
        new_ids: list[str] = []
        new_types: list[str] = []

        if num_boxes > 0:
            telemetry_ctx = self._telemetry_context(state.camera_id)
            for idx in range(num_boxes):
                c_conf = float(conf[idx]) if idx < len(conf) else 0.99
                if idx < len(tracks):
                    _track_id, event_id, tx1, ty1, tx2, ty2, is_new = tracks[idx]
                else:
                    is_new = True
                    event_id = EventLogger.new_event_id()
                    tx1, ty1, tx2, ty2 = (float(v) for v in xyxy[idx])
                if is_new:
                    logged = self._record_new_defect(
                        state, frame, frame_id, event_id, c_conf, (tx1, ty1, tx2, ty2),
                        telemetry_ctx, defect_type=defect_type, severity=int(severity_fn(c_conf)),
                    )
                    if logged:
                        new_ids.append(event_id)
                        new_types.append(defect_type)
            self._process_vibration_flow(state, xyxy)

        self._submit_geometry_check(state, frame)
        return new_ids, new_types

    def _record_new_defect(
        self,
        state: CameraWorkerState,
        frame: np.ndarray,
        frame_id: int,
        event_id: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        telemetry_ctx,
        *,
        defect_type: str,
        severity: int,
        overlay: np.ndarray | None = None,
    ) -> bool:
        # The defect snapshot is the evidence image referenced by the event record.
        evidence_path, overlay_path, crop_path = "", None, None
        h, w = frame.shape[:2]
        x1_c = int(np.clip(bbox[0], 0, w - 1))
        y1_c = int(np.clip(bbox[1], 0, h - 1))
        x2_c = int(np.clip(bbox[2], 0, w))
        y2_c = int(np.clip(bbox[3], 0, h))
        crop_matrix = frame[y1_c:y2_c, x1_c:x2_c].copy()
        if self.disk_monitor.allow_defect_save():
            base = self.defect_dir / state.camera_id / f"{state.camera_id}_defect_{event_id}_frame_{frame_id:04d}"
            if self.enqueue_disk_write(base.with_suffix(".png"), frame):
                evidence_path = str(base.with_suffix(".png").resolve())
            if overlay is not None and self.enqueue_disk_write(Path(f"{base}_overlay.png"), overlay):
                overlay_path = str(Path(f"{base}_overlay.png").resolve())
            if crop_matrix.size and self.enqueue_disk_write(Path(f"{base}_crop.png"), crop_matrix):
                crop_path = str(Path(f"{base}_crop.png").resolve())

        event = DefectEvent(
            event_id=event_id,
            timestamp=EventLogger.now(),
            source=state.camera_id,
            defect_type=defect_type,
            confidence=confidence,
            severity=severity,
            batch_id=None,
            operator_id=None,
            machine_id=telemetry_ctx.machine_id,
            shift_id=telemetry_ctx.shift_id,
            line_id=telemetry_ctx.line_id,
            station_id=telemetry_ctx.station_id,
            temperature=telemetry_ctx.temperature,
            vibration=telemetry_ctx.vibration,
            speed=telemetry_ctx.speed,
            lot_id=telemetry_ctx.lot_id,
            full_image_path=evidence_path,
            overlay_path=overlay_path,
            crop_path=crop_path,
        )

        event_logger = FactoryManager().get_instance("EventLogger")
        if not self._dispatch_event(event_logger, event):
            return False
        state.events_logged += 1
        state.last_log_ts = time.time()
        self._submit_vlm_reasoning(crop_matrix, event_id, event_logger)
        return True

    # ------------------------------------------------------------------
    # Per-part inspection
    # ------------------------------------------------------------------

    def _track_part(self, state: CameraWorkerState, frame: np.ndarray, frame_id: int,
                    event_ids: list[str], defect_types: list[str]) -> None:
        if not self.inspection_enabled:
            return
        tracker = self.part_trackers.get(state.camera_id)
        if tracker is None:
            tracker = self.part_trackers.setdefault(state.camera_id, PartPresenceTracker(
                state.camera_id, on_frames=self._presence_on_frames, off_frames=self._presence_off_frames,
            ))
        present = is_part_present(frame, self.presence_std_threshold)
        completed = tracker.update(frame, present, time.time(), frame_id, event_ids, defect_types)
        if completed is not None:
            try:
                self.inspection_executor.submit(self._inspect_part, state, completed)
            except RuntimeError:
                pass  # shutting down

    def _inspect_part(self, state: CameraWorkerState, part: CompletedPart) -> None:
        """Final verdict for one physical part, recorded against its producing cycle.

        Decision stack (per recipe): good-parts-only anomaly model (catches anything unusual)
        + supervised segmenter (names and delineates known defect types) + frame-level
        detections seen while the part was in view. Any of them failing the part fails it.
        A shadow stack, when registered, judges the same part without affecting the verdict.
        """
        cycle_id = part_key = None
        try:
            frame = part.best_frame
            service = self.process_service
            if service is not None and service.aligner.link_for(state.camera_id) is None:
                service = None  # camera not linked to a machine: no process label to record
            alignment = service.aligner.align(state.camera_id, part.ts) if service is not None else None
            cycle_id = part_key = alignment.cycle_id if alignment else None
            if cycle_id is not None and service.aligner.link_for(state.camera_id).parts_per_cycle > 1:
                part_key = f"{cycle_id}#{alignment.cavity}"   # each cavity is its own part
            recipe = self._recipe_for(state.camera_id, cycle_id)
            stack, shadow = self.stack_provider.get(recipe)

            judged = stack.judge(frame, frame_defect_types=list(part.defect_types or []))
            failed = judged.failed or bool(part.defect_event_ids)
            anomaly, top = judged.anomaly, judged.top
            defect_type, bbox, confidence = judged.defect_type, judged.bbox, judged.confidence
            event_id = part.defect_event_ids[0] if part.defect_event_ids else None
            h, w = frame.shape[:2]

            if failed and event_id is None:
                # No frame-level event exists for this part: log one with part-level evidence.
                event_id = EventLogger.new_event_id()
                overlay = None
                if top is not None and top.mask is not None:
                    overlay = frame.copy()
                    overlay[top.mask] = (0.5 * overlay[top.mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
                    cv2.rectangle(overlay, top.bbox[:2], top.bbox[2:], (0, 0, 255), 2)
                elif anomaly is not None and anomaly.is_anomalous:
                    overlay = stack.anomaly.heatmap_overlay(frame, anomaly, stack.anomaly.threshold)
                severity = self._part_severity(anomaly, top, confidence)
                ctx = self._telemetry_context(state.camera_id, part.ts, cycle_id)
                if not self._record_new_defect(
                    state, frame, part.best_frame_id, event_id, float(confidence or 0.5), bbox or (0, 0, w, h), ctx,
                    defect_type=defect_type or "anomaly", severity=severity, overlay=overlay,
                ):
                    event_id = None

            # No loaded model means nobody looked at this part, which is not a pass.
            not_inspected = stack.empty and not failed
            verdict = "not_inspected" if not_inspected else ("fail" if failed else "pass")
            normalized = judged.normalized_score
            self._inspection_faulted = not_inspected
            if failed:
                self._report_view(state, part_key, self._part_severity(anomaly, top, confidence))
            elif not_inspected:
                self._report_view(state, part_key, PLC_FLAG_NOT_INSPECTED)
                if not state.not_inspected_logged:
                    state.not_inspected_logged = True
                    logger.error("[%s] No inspection model is loaded for recipe '%s': parts are recorded as "
                                 "not_inspected until one is available.", state.camera_id, recipe)
            else:
                self._report_view(state, part_key, PLC_PASS)
            image_sha = None
            if self.audit_log is not None or shadow is not None or self.review_queue is not None:
                from src.inference.audit_log import image_sha256

                image_sha = image_sha256(frame)

            shadow_record = None
            if shadow is not None and not shadow.empty:
                try:
                    sj = shadow.judge(frame, frame_defect_types=list(part.defect_types or []))
                    shadow_record = {
                        "ts": part.ts, "camera_id": state.camera_id, "cycle_id": cycle_id, "image_sha": image_sha,
                        "active_version": stack.versions.get("anomaly", {}).get("version"),
                        "shadow_version": shadow.versions.get("anomaly", {}).get("version"),
                        "active_verdict": verdict, "shadow_verdict": "fail" if (sj.failed or part.defect_event_ids) else "pass",
                        "active_score": normalized, "shadow_score": sj.normalized_score,
                        "active_type": defect_type, "shadow_type": sj.defect_type,
                    }
                    if self.stack_provider.registry is not None:
                        self.stack_provider.registry.log_shadow(recipe, "anomaly", shadow_record)
                except Exception as e:
                    logger.warning("[%s] Shadow model failed on a part: %s", state.camera_id, e)

            self._update_model_health(state.camera_id, recipe, stack, normalized, failed)
            if self.review_queue is not None and self.review_queue.should_review(normalized):
                self.review_queue.add(
                    state.camera_id, frame, normalized_score=normalized, verdict=verdict,
                    cycle_id=cycle_id, event_id=event_id,
                    overlay=stack.anomaly.heatmap_overlay(frame, anomaly, stack.anomaly.threshold),
                    bbox=bbox if failed else (anomaly.bbox if anomaly is not None else None),
                    predicted_type=defect_type, image_sha=image_sha, recipe=recipe,
                )
            elif verdict == "pass":
                pool = self._good_pool(recipe)
                if pool is not None:
                    pool.offer(frame, verdict=verdict, normalized_score=normalized, camera_id=state.camera_id,
                               image_sha=image_sha)

            if self.audit_log is not None:
                self.audit_log.append({
                    "ts": part.ts, "camera_id": state.camera_id, "recipe": recipe, "cycle_id": cycle_id,
                    "verdict": verdict, "defect_type": defect_type if failed else None,
                    "normalized_score": normalized, "raw_score": anomaly.score if anomaly is not None else None,
                    "threshold": stack.anomaly.threshold if stack.anomaly is not None else None,
                    "models": stack.versions, "image_sha256": image_sha, "event_id": event_id,
                    "frame_event_ids": list(part.defect_event_ids or []),
                    "shadow": None if shadow_record is None else {
                        "version": shadow_record["shadow_version"], "verdict": shadow_record["shadow_verdict"],
                        "score": shadow_record["shadow_score"]},
                })

            if not_inspected:
                state.parts_not_inspected += 1
                return
            if service is not None:
                service.record_inspection(
                    state.camera_id, part.ts, verdict,
                    defect_type=defect_type,
                    anomaly_score=normalized,
                    event_id=event_id,
                    alignment=alignment,
                    position=part_pose(frame),
                )
            state.parts_inspected += 1
            if failed:
                state.parts_failed += 1
        except Exception as e:
            logger.error("[%s] Part inspection failed: %s", state.camera_id, e, exc_info=True)
            state.parts_not_inspected += 1
            self._inspection_faulted = True
            self._report_view(state, part_key, PLC_FLAG_NOT_INSPECTED)

    @staticmethod
    def _part_severity(anomaly, top, confidence: float | None) -> int:
        if anomaly is not None and anomaly.is_anomalous and top is None:
            return 3 if anomaly.normalized_score >= 2.0 else (2 if anomaly.normalized_score >= 1.3 else 1)
        return severity_from_conf(confidence or 0.5)

    def _report_view(self, state: CameraWorkerState, part_key: str | None, flag: int) -> None:
        verdict = self.part_fuser.add(part_key, state.camera_id, int(flag), time.time())
        if verdict is not None:
            self._publish_part_verdict(verdict)

    def _publish_part_verdict(self, verdict: dict[str, Any]) -> None:
        """One physical part's final verdict: queued for the PLC and kept in the audit trail."""
        cycle_id = verdict["cycle_id"]
        tail = str(cycle_id).rsplit("-", 1)[-1] if cycle_id else ""
        self._part_counter = (self._part_counter + 1) & 0xFFFF
        part = int(tail) if tail.isdigit() else self._part_counter
        signal = {**verdict, "type": "PART_VERDICT", "part": part & 0xFFFF, "timestamp": time.time()}
        if verdict["verdict"] != "pass" and verdict["missing"]:
            logger.warning("[Inspection] Part %s %s: no result from %s", part, verdict["verdict"], verdict["missing"])
        if self.audit_log is not None:
            self.audit_log.append({"ts": signal["timestamp"], "record": "part_verdict", "part": signal["part"],
                                   "cycle_id": cycle_id, "verdict": verdict["verdict"], "flag": verdict["flag"],
                                   "views": verdict["views"], "missing_views": verdict["missing"]})
        try:
            self.fast_queue.put_nowait(signal)
        except queue.Full:
            logger.error("[Inspection] Reflex queue full: verdict for part %s dropped.", part)

    def _reflex_tick(self, adapter) -> None:
        """Close parts whose views timed out, and keep the PLC heartbeat moving."""
        now = time.time()
        for verdict in self.part_fuser.expired(now):
            self._publish_part_verdict(verdict)
        if adapter is not None and not self._inspection_faulted and now - self._last_heartbeat >= self.heartbeat_s:
            self._last_heartbeat = now
            try:
                self.plc_executor.submit(adapter.write_heartbeat)
            except RuntimeError:
                pass  # shutting down

    def _update_model_health(self, camera_id: str, recipe: str, stack, normalized: float | None, failed: bool) -> None:
        det = stack.anomaly
        version = (stack.versions.get("anomaly") or {}).get("digest")
        key = (camera_id, recipe, version)   # a new model version starts a fresh health baseline
        monitor = self.model_health.get(key)
        if monitor is None:
            from src.inference.anomaly.monitoring import ModelHealthMonitor

            monitor = self.model_health.setdefault(key, ModelHealthMonitor(
                calibration=getattr(det, "calibration", None), threshold=getattr(det, "threshold", None),
                window=int(self._health_cfg.get("window", 300)),
                drift_ratio=float(self._health_cfg.get("drift_ratio", 1.5)),
                expected_reject_rate=float(self._health_cfg.get("expected_reject_rate", 0.05)),
            ))
        status = monitor.update(normalized, failed)
        status["recipe"] = recipe
        status["models"] = {r: v.get("version") for r, v in stack.versions.items()}
        self._health_status = getattr(self, "_health_status", {})
        self._health_status[camera_id] = status
        registry = self.stack_provider.registry
        if (self.retrain_on_drift and registry is not None and det is not None
                and status.get("score_drift") and status["score_drift"] > monitor.drift_ratio):
            registry.request_retrain(recipe, camera_id=camera_id, since_ts=time.time(),
                                     reason=f"score drift {status['score_drift']:.2f}x on {camera_id}")

    def _dispatch_event(self, event_logger, event: DefectEvent) -> bool:
        """Hand a DefectEvent to the EventLogger (async batch path when an event loop is running)."""
        if event_logger is None:
            now = time.time()
            if now - self._last_missing_logger_warning > 5.0:
                logger.warning("[CameraManager] EventLogger not registered; defect events are not being persisted.")
                self._last_missing_logger_warning = now
            return False
        try:
            loop = self.main_loop
            if loop is not None and loop.is_running():
                payload = {k: v for k, v in dataclasses.asdict(event).items() if k != "frame_data"}
                asyncio.run_coroutine_threadsafe(event_logger.log_batch_async([payload]), loop)
            else:
                event_logger.log_event(event)
            return True
        except Exception as e:
            logger.error("[%s] Failed to persist defect event %s: %s", event.source, event.event_id, e)
            return False

    def _submit_vlm_reasoning(self, crop_matrix: np.ndarray, event_id: str, event_logger) -> None:
        """Queue a bounded, non-blocking VLM root-cause pass for a freshly logged defect."""
        if not self.vlm_enabled or crop_matrix.size == 0:
            return
        with self._vlm_lock:
            if self._vlm_pending >= self.vlm_max_pending:
                return  # shed: the VLM backend is slower than the defect rate
            self._vlm_pending += 1

        from src.inference.analysis.vlm_gateway import analyze_defect_crop_sync
        from src.utils.profile_manager import ProfileManager

        def _run():
            try:
                expert_ctx = ProfileManager().get_vlm_expert_context()
                analysis_text = analyze_defect_crop_sync(crop_matrix, expert_ctx, self.config)
                if analysis_text:
                    event_logger.append_vlm_narrative(event_id, analysis_text)
            except Exception as e:
                logger.error("[VLM] Reasoning pass failed for %s: %s", event_id, e)
            finally:
                with self._vlm_lock:
                    self._vlm_pending -= 1

        try:
            self.vlm_executor.submit(_run)
        except RuntimeError:
            with self._vlm_lock:
                self._vlm_pending -= 1

    def _process_vibration_flow(self, state: CameraWorkerState, xyxy: np.ndarray) -> None:
        """Feed the centroid displacement of the dominant detection into the vibration DSP."""
        analyzer = self.vibration_analyzers.get(state.camera_id)
        if analyzer is None or len(xyxy) == 0:
            return
        try:
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            best = int(np.argmax(areas))
            cx = float((xyxy[best, 0] + xyxy[best, 2]) * 0.5)
            cy = float((xyxy[best, 1] + xyxy[best, 3]) * 0.5)
            prev_cx, prev_cy = state.prev_centroid if state.prev_centroid is not None else (cx, cy)
            state.prev_centroid = (cx, cy)
            analyzer.ingest(cx - prev_cx, cy - prev_cy)
        except Exception as e:
            logger.error("[%s] Vibration flow update failed: %s", state.camera_id, e)

    def _submit_geometry_check(self, state: CameraWorkerState, frame: np.ndarray) -> None:
        geom_engine = self.geometry_engines.get(state.camera_id)
        # Without a golden template every check is an ALIGNMENT_FAULT; skip the wasted work.
        if geom_engine is None or geom_engine.template is None:
            return
        # Bounded in-flight work: never let geometry checks queue up behind a slow CPU.
        with self._geometry_lock:
            if state.geometry_inflight >= 2:
                return
            state.geometry_inflight += 1

        def _geom_callback(fut):
            with self._geometry_lock:
                state.geometry_inflight -= 1
            try:
                res = fut.result()
                if not res:
                    return
                state.last_geometric_diagnostic = res
                violation = res.get("status") == "GEOMETRIC_VIOLATION"

                now = time.time()
                if violation or (now - state.last_geometry_log_ts) >= self.geometry_ok_log_interval:
                    state.last_geometry_log_ts = now
                    res_payload = copy.deepcopy(res)
                    res_payload["camera_id"] = state.camera_id
                    res_payload["timestamp"] = now
                    ev_logger = FactoryManager().get_instance("EventLogger")
                    if ev_logger:
                        loop = self.main_loop
                        if loop is not None and loop.is_running():
                            asyncio.run_coroutine_threadsafe(ev_logger.log_batch_async([res_payload]), loop)
                        else:
                            ev_logger._sync_execute_transaction([res_payload])

                if violation:
                    try:
                        self.fast_queue.put_nowait({
                            "type": "GEOMETRIC_CRITICAL",
                            "camera_id": state.camera_id,
                            "timestamp": now,
                            "diagnostic": res,
                        })
                    except queue.Full:
                        pass
            except Exception as e:
                logger.error("[%s] Background geometry check crashed: %s", state.camera_id, e)

        try:
            fut = self.executor.submit(geom_engine.inspect_part_geometry, frame.copy(), False)
            fut.add_done_callback(_geom_callback)
        except RuntimeError:
            with self._geometry_lock:
                state.geometry_inflight -= 1  # executor already shut down

    # ------------------------------------------------------------------
    # Reflex + supervision
    # ------------------------------------------------------------------

    def _reflex_worker(self):
        adapter = getattr(self.telemetry, "adapter", None)
        while not self.shutdown_event.is_set():
            self._reflex_tick(adapter)
            try:
                signal_data = self.fast_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if adapter is not None:
                    write = adapter.write_part_verdict if signal_data.get("type") == "PART_VERDICT" \
                        else adapter.write_advisory_flag
                    try:
                        self.plc_executor.submit(write, signal_data)
                    except RuntimeError:
                        pass

                hub_connector = FactoryManager().get_instance("HubConnector")
                loop = self.main_loop
                if hub_connector is not None and hasattr(hub_connector, "bridge") and loop is not None and loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        hub_connector.bridge.publish_telemetry("qualyxus/reflex/anomaly", signal_data),
                        loop,
                    )
            except Exception as e:
                logger.error("[Reflex] Failed to dispatch reflex signal: %s", e)

    @staticmethod
    def _atomic_write_json(path: Path, payload: dict) -> None:
        tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, default=str)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.debug("[CameraManager] Status export to %s failed: %s", path, e)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _publish_runtime_status(self) -> None:
        """Export the camera status consumed by the healthcheck/dashboard and the guardian vitals."""
        self._atomic_write_json(self.status_path, {"timestamp": time.time(), "workers": self.get_status_snapshot()})
        try:
            state = self._current_policy_state()
            vitals = dataclasses.asdict(self.governor.get_vitals())
        except Exception as e:
            logger.debug("[CameraManager] Governor vitals unavailable: %s", e)
            return
        if state != self._last_policy_state:
            self._resource_log.appendleft(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {state.value}")
            self._last_policy_state = state
        self._atomic_write_json(
            self.system_health_path,
            {"timestamp": time.time(), "state": state.value, "vitals": vitals, "resource_log": list(self._resource_log)},
        )

    def _monitor_loop(self):
        while not self.shutdown_event.wait(self.monitor_interval):
            with self._lock:
                states = list(self.workers.values())

            for state in states:
                if self.shutdown_event.is_set():
                    break
                cam_id = state.camera_id

                dead_thread = state.capture_thread is not None and not state.capture_thread.is_alive()
                if dead_thread and not state.failed:
                    state.failed = True
                    state.failure_reason = "Thread unexpectedly terminated."
                    state.stop_event.set()

                decision = self.watchdog.evaluate(state)

                if decision.restart:
                    logger.warning("[CameraManager] Watchdog enforcing restart on %s. Reason: %s", cam_id, decision.reason)
                    state.failed = True
                    state.stop_event.set()

                    if state.capture_thread and state.capture_thread.is_alive():
                        state.capture_thread.join(timeout=0.5)
                    if state.inference_thread and state.inference_thread.is_alive():
                        state.inference_thread.join(timeout=0.5)

                    cam_cfg = next((c for c in self._camera_entries() if str(c.get("id")) == cam_id), None)
                    if cam_cfg and not self.shutdown_event.is_set():
                        logger.info("[CameraManager] Respawning worker node %s (Attempt %d)...", cam_id, state.restarts + 1)
                        self._start_camera_worker(cam_cfg, previous_restarts=state.restarts + 1)

                elif state.frames_captured > 10 and not state.failed:
                    self.watchdog.record_recovered(cam_id)

            if not self.shutdown_event.is_set():
                self._publish_runtime_status()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting Standalone CameraManager Service Daemon Appliance...")

    manager = CameraManager()
    manager.start()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("Shutdown terminal sequence intercepted. Gracefully severing camera pipelines...")
        manager.stop()
