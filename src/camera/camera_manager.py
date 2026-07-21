import copy
import hashlib
import json
import logging
import os
import queue
import threading
import time
import uuid
import re
import asyncio
import torch

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

logger = logging.getLogger(__name__)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config.config import Config
from src.inference.postprocess import make_preprocessor, overlay_mask, severity_from_conf
from src.logging.event_logger import EventLogger
from src.logging.schema import DefectEvent, SystemState
from src.system.disk_monitor import DiskMonitor
from src.system.resource_monitor import ResourceGovernor
from src.system.runtime_metrics import RuntimeMetricsLogger
from src.telemetry.telemetry_provider import TelemetryProvider
from src.inference.analysis.vibration_engine import VibrationAnalyzer
from src.system.watchdog import WatchdogManager
from src.core.factory_manager import FactoryManager
from concurrent.futures import ThreadPoolExecutor
from src.inference.analysis.geometry_engine import GeometryEngine

class CoercibleAwaitable:
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
                    cls._instance._barrier_lock = threading.Lock()
                    cls._instance._condition = threading.Condition(cls._instance._barrier_lock)
        return cls._instance

    def register_channel(self, camera_id: str):
        with self._barrier_lock:
            self._channels[camera_id] = 0

    def unregister_channel(self, camera_id: str):
        with self._barrier_lock:
            self._channels.pop(camera_id, None)
            self._condition.notify_all()

    def sync_barrier(self, camera_id: str, sequence_id: int, timeout_sec: float = 0.05) -> bool:
        with self._barrier_lock:
            if camera_id not in self._channels:
                return True
            self._channels[camera_id] = sequence_id
            
            all_synced = all(seq >= sequence_id for seq in self._channels.values())
            if all_synced:
                self._condition.notify_all()
                return True

            start_t = time.perf_counter()
            while not all(seq >= sequence_id for seq in self._channels.values()):
                rem = timeout_sec - (time.perf_counter() - start_t)
                if rem <= 0:
                    return False
                self._condition.wait(timeout=min(0.005, rem))
            return True


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
    tracker: Any = None
    pool: list = field(default_factory=list)
    pool_write_index: int = 0

class CameraManager:
    """
    High-Velocity Ingestion Pipeline.
    Strictly handles RTSP/USB capture, Triton preprocessing, and async delegation.
    """
    def __init__(self, config_path: str = "config.yaml", config_data: dict | None = None):
        self.config_path = config_path
        self.raw_config = self._load_raw_config(config_path, config_data)
        self.config = Config(raw=copy.deepcopy(self.raw_config))

        self.shutdown_event = threading.Event()
        self.fast_queue: queue.Queue = queue.Queue(maxsize=20)
        self.workers: dict[str, CameraWorkerState] = {}
        
        telemetry_mode = str(self.config.get("telemetry", "mode", default="simulated"))
        self.telemetry = TelemetryProvider(
            simulation_mode=telemetry_mode == "simulated",
            mode=telemetry_mode,
            protocol=str(self.config.get("telemetry", "protocol", default="modbus")),
        )

        self.governor = ResourceGovernor(config=self.config)
        FactoryManager().register_instance("ResourceGovernor", self.governor)

        self.frame_skip = 1

        self.monitor_interval = float(self.raw_config.get("camera_manager", {}).get("monitor_interval", 2.0))
        self.restart_backoff = float(self.raw_config.get("camera_manager", {}).get("restart_backoff", 1.5))

        hardening_cfg = self.raw_config.get("hardening", {})
        self.max_pending_events = int(hardening_cfg.get("max_pending_events", 100))
        self.max_events_per_sec = float(hardening_cfg.get("max_events_per_sec", 10.0))

        test_cfg = self.raw_config.get("test_flags", {})
        self.simulate_camera_drop = bool(test_cfg.get("simulate_camera_drop", False))

        self._lock = threading.Lock()
        self._monitor_thread: threading.Thread | None = None

        watchdog_cfg = self.raw_config.get("watchdog", {})
        self.watchdog = WatchdogManager(
            heartbeat_timeout=float(watchdog_cfg.get("heartbeat_timeout", 10.0)),
            frame_drop_threshold=int(watchdog_cfg.get("frame_drop_threshold", 200)),
            queue_overflow_streak_threshold=int(watchdog_cfg.get("queue_overflow_streak_threshold", 25)),
            base_backoff=float(watchdog_cfg.get("base_backoff", 1.0)),
            max_backoff=float(watchdog_cfg.get("max_backoff", 30.0)),
        )

        disk_cfg = self.raw_config.get("disk_monitor", {})
        self.disk_monitor = DiskMonitor(
            watch_path=disk_cfg.get("watch_path", "data"),
            interval_seconds=float(disk_cfg.get("interval_seconds", 60.0)),
        )

        self.vibration_analyzers: dict[str, VibrationAnalyzer] = {}
        self.geometry_engines: dict[str, GeometryEngine] = {}
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="geometry-worker")
        self.vlm_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-reasoner")
        vibration_cfg = self.raw_config.get("vibration_analysis", {}) or {}
        self._kap_db_path: str = str(vibration_cfg.get("kap_db_path", "data/knowledge/knowledge_base.db"))
        
        self.main_loop = None

        from src.inference.multiplexer import InferenceMultiplexer
        self.multiplexer = InferenceMultiplexer(config=self.config)
        FactoryManager().register_instance("InferenceMultiplexer", self.multiplexer)

        if not FactoryManager().get_instance("EventLogger"):
            logging_cfg = self.raw_config.get("logging", {}) or {}
            base_sqlite = logging_cfg.get("sqlite_path", "data/events/events.db")
            base_csv = logging_cfg.get("csv_path", "data/events/events.csv")
            event_logger = EventLogger(sqlite_path=base_sqlite, csv_path=base_csv)
            FactoryManager().register_instance("EventLogger", event_logger)
        FactoryManager().register_instance("CameraManager", self)

    @staticmethod
    def _load_raw_config(config_path: str, config_data: dict | None) -> dict:
        if config_data is not None:
            return copy.deepcopy(config_data)
        return Config.load(config_path=config_path).raw

    def _global_camera_defaults(self) -> dict:
        return self.raw_config.get("camera", {})

    def _camera_entries(self) -> list[dict]:
        cameras = self.raw_config.get("cameras", [])
        if not cameras:
            raise ValueError("No cameras configured.")
        return [c for c in cameras if c.get("enabled", True)]

    def _make_session_id(self, camera_id: str) -> str:
        return f"{camera_id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    def _start_camera_worker(self, camera_cfg: dict, previous_restarts: int = 0):
        camera_id = str(camera_cfg.get("id", "camera"))

        use_synth_env = os.getenv("QUALYXUS_USE_SYNTHETIC_CAMERAS", "0").strip().lower() in {"1", "true", "yes"}
        force_mock_env = os.getenv("QUALYXUS_FORCE_MOCK_INFERENCE", "0").strip().lower() in {"1", "true", "yes"}

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
            frame_queue=queue.Queue(maxsize=16),
            event_queue=queue.Queue(maxsize=64),
            stop_event=threading.Event(),
            restarts=previous_restarts,
        )

        state.capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"capture-{camera_id}",
            args=(state, camera_cfg),
            daemon=True,
        )
        
        mock_inference = bool(camera_cfg.get("mock_inference", False))
        if mock_inference or force_mock_env:
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

        self.vibration_analyzers[camera_id] = VibrationAnalyzer(
            buffer_size=512,
            kap_db_path=self._kap_db_path,
            fast_queue=self.fast_queue,
            camera_id=camera_id,
        )

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
        self.disk_monitor.start()

        self._reflex_thread = threading.Thread(target=self._reflex_worker, daemon=True)
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
        logger.info("[CameraManager] Ingestion pipeline started")
        return CoercibleAwaitable(None)

    def stop_all_streams(self) -> None:
        """Synchronous termination of all active camera ingestion threads."""
        self._sync_stop()

    def _sync_stop(self):
        self.shutdown_event.set()
        
        if hasattr(self, "multiplexer"):
            self.multiplexer.stop()

        for t in [getattr(self, "_reflex_thread", None)]:
            if t is not None and t.is_alive():
                t.join(timeout=1.0)

        with self._lock:
            states = list(self.workers.values())

        for state in states:
            state.stop_event.set()

        for state in states:
            for t in (state.capture_thread, state.inference_thread):
                if t is not None and t.is_alive():
                    t.join(timeout=5.0)

        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)

        self.executor.shutdown(wait=False)
        self.vlm_executor.shutdown(wait=False)
        self.disk_monitor.stop()

    def stop(self) -> CoercibleAwaitable:
        self.stop_all_streams()
        logger.info("[CameraManager] Ingestion pipeline stopped")
        return CoercibleAwaitable(None)

    def get_status_snapshot(self) -> dict[str, dict[str, Any]]:
        """Fetch active processing metrics and lifecycle status across all camera streams."""
        with self._lock:
            out = {}
            for camera_id, state in self.workers.items():
                capture_alive = state.capture_thread is not None and state.capture_thread.is_alive()
                logger_alive = False
                
                event_logger = FactoryManager().get_instance("EventLogger")
                if event_logger:
                    if getattr(event_logger, "worker_task", None) is not None:
                        logger_alive = not event_logger.worker_task.done()
                    else:
                        logger_alive = getattr(event_logger, "_initialized", False)
                
                out[camera_id] = {
                    "session_id": state.session_id,
                    "failed": state.failed,
                    "failure_reason": state.failure_reason,
                    "frames_captured": state.frames_captured,
                    "frames_inferred": state.frames_inferred,
                    "events_logged": state.events_logged,
                    "frame_queue_size": state.frame_queue.qsize(),
                    "event_queue_size": state.event_queue.qsize(),
                    "last_capture_ts": state.last_capture_ts,
                    "last_infer_ts": state.last_infer_ts,
                    "last_log_ts": getattr(state, "last_log_ts", 0.0),
                    "restarts": state.restarts,
                    "reconnect_count": state.reconnect_count,
                    "dropped_frames": state.dropped_frames,
                    "backpressure_warnings": state.backpressure_warnings,
                    "queue_overflow_streak": state.queue_overflow_streak,
                    "last_heartbeat_ts": state.last_heartbeat_ts,
                    "started_at": state.started_at,
                    "capture_alive": capture_alive,
                    "inference_alive": bool(state.inference_thread and state.inference_thread.is_alive()),
                    "logger_alive": logger_alive,
                    "last_geometric_diagnostic": getattr(state, "last_geometric_diagnostic", {}),
                }
            return out

    def _open_camera(self, source: Any, width: int = 640, height: int = 480, fps: float = 30.0):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            return (cap, "OpenCV-HW" if str(source).startswith("rtsp://") else "OpenCV-SW")
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

    def _capture_loop(self, state: CameraWorkerState, camera_cfg: dict):
        defaults = self._global_camera_defaults()
        width = int(camera_cfg.get("width", defaults.get("width", 640)))
        height = int(camera_cfg.get("height", defaults.get("height", 480)))
        fps = float(camera_cfg.get("fps", defaults.get("fps", 20)))
        reconnect_delay = 5.0
        
        accel_cfg = self.raw_config.get("hardware_acceleration", {}) or {}
        use_triton = bool(accel_cfg.get("use_triton", True))
        precision = str(accel_cfg.get("precision", "fp16"))
        local_preprocessor = make_preprocessor(use_triton=use_triton, precision=precision)
        
        is_synthetic = isinstance(state.source, str) and state.source.startswith("synthetic://")
        mock_inference_cfg = bool(camera_cfg.get("mock_inference", False))
        
        # Register channel with multi-camera sync barrier
        sync_mgr = CameraSyncManager()
        sync_mgr.register_channel(state.camera_id)

        # Pre-allocate 128-slot mutable ring buffer array managed within CameraWorkerState
        state.pool = [FrameSlot(height, width) for _ in range(128)]
        state.pool_write_index = 0

        cap = None
        if not is_synthetic:
            cap, state.ingest_mode = self._open_camera(state.source, width=width, height=height, fps=fps)
        else:
            state.ingest_mode = "Synthetic"

        frame_interval = 1.0 / max(1.0, fps)
        frame_id = 0
        manager = FactoryManager()

        try:
            while not self.shutdown_event.is_set() and not state.stop_event.is_set():
                
                if self.governor and self.governor.get_policy_state() == SystemState.CRITICAL:
                    current_ts = time.time()
                    if current_ts - state.last_throttle_log_ts > 1.0:
                        logger.warning("[%s] Governor STATE_CRITICAL: Memory saturated. Aggressive shedding active.", state.camera_id)
                        state.last_throttle_log_ts = current_ts
                        
                    if cap is not None and not is_synthetic:
                        cap.grab()
                    
                    state.last_heartbeat_ts = time.perf_counter()
                    self._interruptible_sleep(0.1, state)
                    continue

                loop_start = time.perf_counter()

                if is_synthetic:
                    synthetic_fail_after = camera_cfg.get("synthetic_fail_after_frames")
                    if synthetic_fail_after is not None and state.restarts == 0 and frame_id >= int(synthetic_fail_after):
                         raise RuntimeError("Synthetic capture failure injected")
                    slot = state.pool[state.pool_write_index % 128]
                    slot.frame[:] = (40, 40, 40)
                    frame = slot.frame
                    state.pool_write_index += 1
                    ret = True
                else:
                    if cap is None:
                        if not self._interruptible_sleep(reconnect_delay, state):
                            break
                        cap, _ = self._open_camera(state.source, width=width, height=height, fps=fps)
                        if cap is None:
                            continue
                        state.reconnect_count += 1

                    grabbed = cap.grab()
                    sync_mgr.sync_barrier(state.camera_id, frame_id, timeout_sec=0.05)
                    if grabbed:
                        slot = state.pool[state.pool_write_index % 128]
                        ret, decoded_img = cap.retrieve(image=slot.frame)
                        if not ret or (decoded_img is not slot.frame):
                            logger.warning("[%s] OpenCV bypassed pre-allocated buffer allocations. Strides or alignment mismatch.", state.camera_id)
                            frame = decoded_img
                        else:
                            frame = slot.frame
                        state.pool_write_index += 1
                        ret = True
                    else:
                        ret = False

                    if not ret:
                        cap.release()
                        cap = None
                        if not self._interruptible_sleep(reconnect_delay, state):
                            break
                        continue

                    if not ret:
                        cap.release()
                        cap = None
                        if not self._interruptible_sleep(reconnect_delay, state):
                            break
                        continue

                frame_id += 1
                state.frames_captured = frame_id
                state.last_capture_ts = time.perf_counter()
                state.last_heartbeat_ts = state.last_capture_ts
                timestamp_perf = time.perf_counter()

                try:
                    raw_out_dir = Path("data/live/raw")
                    raw_out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(raw_out_dir / f"{state.camera_id}_frame_{frame_id:04d}.png"), frame)
                except Exception as disk_err:
                    logger.error("Failed to persist unique frame capture layout: %s", disk_err)

                if not (mock_inference_cfg or state.force_mock_inference):
                    try:
                        multiplexer = manager.get_instance("InferenceMultiplexer")
                        event_logger = manager.get_instance("EventLogger")
                        
                        result = None
                        if multiplexer:
                            try:
                                request_id = multiplexer.submit_frame(state.camera_id, frame, block=False, timeout=None)
                                result = multiplexer.retrieve_result(request_id, timeout_sec=0.033)
                            except queue.Full:
                                result = None
                        
                        if result is None:
                            if not is_synthetic:
                                state.dropped_frames += 1
                            elapsed = time.perf_counter() - loop_start
                            if elapsed < frame_interval:
                                self._interruptible_sleep(frame_interval - elapsed, state)
                            continue

                        boxes, masks, xyxy, conf = result
                        state.frames_inferred += 1
                        state.last_infer_ts = time.perf_counter()
                        
                        num_boxes = len(boxes)
                        detections = np.empty((num_boxes, 6), dtype=np.float32)
                        if num_boxes > 0:
                            detections[:, :4] = xyxy
                            detections[:, 4] = conf
                            detections[:, 5] = boxes[:, 5] if boxes.shape[1] > 5 else 0.0
                            
                        tracks = []
                        if state.tracker is not None:
                            tracks = state.tracker.update(detections)
                        
                        if len(boxes) > 0:
                            telemetry_ctx = self.telemetry.get_telemetry(state.camera_id)
                            for idx, box in enumerate(boxes):
                                c_conf = float(conf[idx]) if idx < len(conf) else 0.99
                                
                                if idx < len(tracks):
                                    track_id, event_id, tx1, ty1, tx2, ty2, is_new = tracks[idx]
                                else:
                                    is_new = True
                                    event_id = EventLogger.new_event_id()
                                    tx1, ty1, tx2, ty2 = xyxy[idx]
                                    
                                event = DefectEvent(
                                    event_id=event_id,
                                    timestamp=EventLogger.now(),
                                    source=state.camera_id,
                                    defect_type="anomaly",
                                    confidence=c_conf,
                                    severity=severity_from_conf(c_conf),
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
                                    full_image_path="",
                                    overlay_path=None,
                                    crop_path=None,
                                )
                                
                                if is_new:
                                    if event_logger:
                                        if self.main_loop:
                                            import dataclasses
                                            if dataclasses.is_dataclass(event):
                                                payload = {k: v for k, v in dataclasses.asdict(event).items() if k != "frame_data"}
                                        elif hasattr(event, "model_dump"):
                                            payload = event.model_dump()
                                        else:
                                            payload = event.dict()
                                        asyncio.run_coroutine_threadsafe(event_logger.log_batch_async([payload]), self.main_loop)
                                    else:
                                        event_logger.log_event(event)
                                    state.events_logged += 1

                                    try:
                                        defect_out_dir = Path("data/live/defects")
                                        defect_out_dir.mkdir(parents=True, exist_ok=True)
                                        cv2.imwrite(str(defect_out_dir / f"{state.camera_id}_defect_{event_id}_frame_{frame_id:04d}.png"), frame)
                                    except Exception as disk_err:
                                        logger.error("Failed to write unique defect sequence matrix: %s", disk_err)

                                    h, w = frame.shape[:2]
                                    x1_c = int(np.clip(tx1, 0, w - 1))
                                    y1_c = int(np.clip(ty1, 0, h - 1))
                                    x2_c = int(np.clip(tx2, 0, w))
                                    y2_c = int(np.clip(ty2, 0, h))
                                    crop_matrix = frame[y1_c:y2_c, x1_c:x2_c].copy()

                                    from src.utils.profile_manager import ProfileManager
                                    expert_ctx = ProfileManager().get_vlm_expert_context()

                                    from src.inference.analysis.vlm_gateway import analyze_defect_crop_sync
                                    def execute_vlm_decoupled_pass(c_mat, e_ctx, ev_id, config_obj, logger_obj):
                                        analysis_text = analyze_defect_crop_sync(c_mat, e_ctx, config_obj)
                                        logger_obj.append_vlm_narrative(ev_id, analysis_text)

                                    self.vlm_executor.submit(execute_vlm_decoupled_pass, crop_matrix, expert_ctx, event_id, self.config, event_logger)
                                
                            if multiplexer:
                                multiplexer._process_vibration_flow(self, state.camera_id, xyxy)
                                
                        geom_engine = self.geometry_engines.get(state.camera_id)
                        if geom_engine:
                            def _geom_callback(fut):
                                try:
                                    res = fut.result()
                                    state.last_geometric_diagnostic = res
                                    
                                    if res:
                                        res_payload = copy.deepcopy(res)
                                        res_payload["camera_id"] = state.camera_id
                                        res_payload["timestamp"] = time.time()
                                        ev_logger = FactoryManager().get_instance("EventLogger")
                                        if ev_logger:
                                            if self.main_loop:
                                                asyncio.run_coroutine_threadsafe(
                                                    ev_logger.log_batch_async([res_payload]),
                                                    self.main_loop
                                                )
                                            else:
                                                ev_logger._sync_execute_transaction([res_payload])
                                                
                                    if res and res.get("status") == "GEOMETRIC_VIOLATION":
                                        try:
                                            self.fast_queue.put_nowait({
                                                "type": "GEOMETRIC_CRITICAL",
                                                "camera_id": state.camera_id,
                                                "timestamp": time.time(),
                                                "diagnostic": res
                                            })
                                        except queue.Full:
                                            pass
                                except Exception as e:
                                    logger.error("[%s] Background geometry check crashed: %s", state.camera_id, e)
                            try:
                                fut = self.executor.submit(geom_engine.inspect_part_geometry, frame, False)
                                fut.add_done_callback(_geom_callback)
                            except RuntimeError:
                                pass
                    except Exception as e:
                        logger.error("[%s] Multiplexer handoff failed: %s", state.camera_id, e)
                else:
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
                    except queue.Full:
                        if not is_synthetic:
                            state.dropped_frames += 1

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

    def _mock_inference_loop(self, state: CameraWorkerState, camera_cfg: dict):
        mock_event_every = 10
        manager = FactoryManager()
        event_logger = manager.get_instance("EventLogger")
        multiplexer = manager.get_instance("InferenceMultiplexer")
        
        try:
            while True:
                if (self.shutdown_event.is_set() or state.stop_event.is_set()) and state.frame_queue.empty():
                    break
                try:
                    packet: FramePacket = state.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                try:
                    raw_out_dir = Path("data/live/raw")
                    raw_out_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(raw_out_dir / f"{state.camera_id}_frame_{packet.frame_id:04d}.png"), packet.frame)
                except Exception as disk_err:
                    logger.error("Failed to write sequential image during mock pipeline run: %s", disk_err)

                if multiplexer:
                    try:
                        request_id = multiplexer.submit_frame(state.camera_id, packet.frame)
                        result = multiplexer.retrieve_result(request_id, timeout_sec=1.0)
                    except queue.Full:
                        result = None
                        
                    if result is None:
                        is_synthetic = isinstance(state.source, str) and state.source.startswith("synthetic://")
                        if not is_synthetic:
                            state.dropped_frames += 1
                        self._interruptible_sleep(0.01, state)
                        continue
                     
                    boxes, masks, xyxy, conf = result
                    state.frames_inferred += 1
                    state.last_infer_ts = time.perf_counter()

                    if len(boxes) == 0 and packet.frame_id % max(1, mock_event_every) == 0:
                        offset = (packet.frame_id % 10) * 100.0
                        x1 = float(10.0 + offset)
                        y1 = float(10.0)
                        x2 = float(90.0 + offset)
                        y2 = float(90.0)
                        boxes = np.array([[x1, y1, x2, y2, 0.99, 0.0]], dtype=np.float32)
                        xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)
                        conf = np.array([0.99], dtype=np.float32)
                        
                    num_boxes = len(boxes)
                    detections = np.empty((num_boxes, 6), dtype=np.float32)
                    if num_boxes > 0:
                        detections[:, :4] = xyxy
                        detections[:, 4] = conf
                        detections[:, 5] = boxes[:, 5] if boxes.shape[1] > 5 else 0.0
                        
                    tracks = []
                    if state.tracker is not None:
                        tracks = state.tracker.update(detections)

                    geom_engine = self.geometry_engines.get(state.camera_id)
                    if geom_engine:
                        def _geom_callback(fut):
                            try:
                                res = fut.result()
                                state.last_geometric_diagnostic = res
                                
                                if res:
                                    res_payload = copy.deepcopy(res)
                                    res_payload["camera_id"] = state.camera_id
                                    res_payload["timestamp"] = time.time()
                                    ev_logger = FactoryManager().get_instance("EventLogger")
                                    if ev_logger:
                                        if self.main_loop:
                                            asyncio.run_coroutine_threadsafe(
                                                ev_logger.log_batch_async([res_payload]),
                                                self.main_loop
                                            )
                                        else:
                                            ev_logger._sync_execute_transaction([res_payload])
                                        
                                if res and res.get("status") == "GEOMETRIC_VIOLATION":
                                    try:
                                        self.fast_queue.put_nowait({
                                            "type": "GEOMETRIC_CRITICAL",
                                            "camera_id": state.camera_id,
                                            "timestamp": time.time(),
                                            "diagnostic": res
                                        })
                                    except queue.Full:
                                        pass
                            except Exception as e:
                                logger.error("[%s] Mock background geometry check crashed: %s", state.camera_id, e)
                        try:
                            fut = self.executor.submit(geom_engine.inspect_part_geometry, packet.frame, False)
                            fut.add_done_callback(_geom_callback)
                        except RuntimeError:
                            pass

                    if len(boxes) > 0:
                        telemetry_ctx = self.telemetry.get_telemetry(state.camera_id)
                        for idx, box in enumerate(boxes):
                            c_conf = float(conf[idx]) if idx < len(conf) else 0.99
                            if idx < len(tracks):
                                track_id, event_id, tx1, ty1, tx2, ty2, is_new = tracks[idx]
                            else:
                                is_new = True
                                event_id = EventLogger.new_event_id()
                                tx1, ty1, tx2, ty2 = xyxy[idx]
                                
                            event = DefectEvent(
                                event_id=event_id,
                                timestamp=EventLogger.now(),
                                source=state.camera_id,
                                defect_type="mock_defect",
                                confidence=c_conf,
                                severity=3,
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
                                full_image_path="",
                                overlay_path=None,
                                crop_path=None,
                            )
                            
                            if is_new:
                                if event_logger:
                                    if self.main_loop:
                                        import dataclasses
                                        if dataclasses.is_dataclass(event):
                                            payload = {k: v for k, v in dataclasses.asdict(event).items() if k != "frame_data"}
                                        elif hasattr(event, "model_dump"):
                                            payload = event.model_dump()
                                        else:
                                            payload = event.dict()
                                        asyncio.run_coroutine_threadsafe(event_logger.log_batch_async([payload]), self.main_loop)
                                    else:
                                        event_logger.log_event(event)
                                    state.events_logged += 1

                                try:
                                    defect_out_dir = Path("data/live/defects")
                                    defect_out_dir.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(defect_out_dir / f"{state.camera_id}_defect_{event_id}_frame_{packet.frame_id:04d}.png"), packet.frame)
                                except Exception as disk_err:
                                    logger.error("Failed to write mock defect snapshot to drive layout: %s", disk_err)

                                h, w = packet.frame.shape[:2]
                                x1_c = int(np.clip(tx1, 0, w - 1))
                                y1_c = int(np.clip(ty1, 0, h - 1))
                                x2_c = int(np.clip(tx2, 0, w))
                                y2_c = int(np.clip(ty2, 0, h))
                                crop_matrix = packet.frame[y1_c:y2_c, x1_c:x2_c].copy()

                                from src.utils.profile_manager import ProfileManager
                                expert_ctx = ProfileManager().get_vlm_expert_context()

                                from src.inference.analysis.vlm_gateway import analyze_defect_crop_sync
                                def execute_vlm_decoupled_pass(c_mat, e_ctx, ev_id, config_obj, logger_obj):
                                    analysis_text = analyze_defect_crop_sync(c_mat, e_ctx, config_obj)
                                    logger_obj.append_vlm_narrative(ev_id, analysis_text)

                                self.vlm_executor.submit(execute_vlm_decoupled_pass, crop_matrix, expert_ctx, event_id, self.config, event_logger)
                else:
                    self._interruptible_sleep(0.01, state)
                    
        except Exception as e:
            state.failed = True
            state.failure_reason = str(e)
            state.stop_event.set()

    def _reflex_worker(self):
        hub_connector = FactoryManager().get_instance("HubConnector")
        while not self.shutdown_event.is_set():
            try:
                signal_data = self.fast_queue.get(timeout=0.5)
                
                if hasattr(self, "telemetry") and self.telemetry is not None:
                    if hasattr(self.telemetry, "adapter") and self.telemetry.adapter is not None:
                        try:
                            self.vlm_executor.submit(self.telemetry.adapter.write_anomaly_interlock, signal_data)
                        except RuntimeError:
                            pass
                
                if hub_connector and hasattr(hub_connector, "bridge") and self.main_loop:
                    asyncio.run_coroutine_threadsafe(
                        hub_connector.bridge.publish_telemetry("qualxyus/reflex/anomaly", signal_data),
                        self.main_loop
                    )
            except queue.Empty:
                pass
            except Exception:
                pass

    def _monitor_loop(self):
        while not self.shutdown_event.is_set():
            time.sleep(self.monitor_interval)
            if self.shutdown_event.is_set():
                break
            with self._lock:
                states = list(self.workers.values())
             
            for state in states:
                if self.shutdown_event.is_set():
                    break
                cam_id = state.camera_id
                
                dead_thread = state.capture_thread is not None and not state.capture_thread.is_alive()
                if dead_thread and not state.failed:
                    if self.shutdown_event.is_set():
                        break
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
                    if cam_cfg:
                        logger.info("[CameraManager] Respawning worker node %s (Attempt %d)...", cam_id, state.restarts + 1)
                        self._start_camera_worker(cam_cfg, previous_restarts=state.restarts + 1)
                        
                elif getattr(state, "frames_captured", 0) > 10:
                    self.watchdog.record_recovered(cam_id)

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