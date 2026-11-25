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
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil as _psutil
except ImportError:
    _psutil = None

from src.camera.trigger import PartTrigger, TriggerSource
from src.camera.view_store import ViewStore
from src.camera.presence import PresenceDetector
from src.camera.part_inspection import PartPresenceTracker, PartVerdictFuser
from src.config.schema import check_real_plc, required_views
from src.config.config import Config
from src.core.factory_manager import FactoryManager
from src.inference.analysis.geometry_engine import GeometryEngine
from src.inference.analysis.vibration_engine import VibrationAnalyzer
from src.logging.event_logger import EventLogger
from src.logging.schema import SystemState
from src.system.disk_monitor import DiskMonitor
from src.system.resource_monitor import ResourceGovernor
from src.system.runtime_metrics import RuntimeMetricsLogger
from src.system.watchdog import WatchdogManager
from src.telemetry.telemetry_provider import TelemetryProvider

from src.camera.capture import CaptureMixin
from src.camera.defects import DefectMixin
from src.camera.inspection import InspectionMixin
from src.camera.verdicts import VerdictMixin
from src.camera.worker import (  # noqa: F401 - re-exported
    _env_flag, CoercibleAwaitable, FramePacket, FrameSlot, pool_slots, CameraWorkerState, _TRUTHY, _SHEDDING_STATES, FRAME_POOL_BYTES, BACKLOG_BYTES, INSPECT_BATCH, EVIDENCE_PENDING, VERDICT_LOST_HOLD_S, UNKNOWN_RECIPE, UNSEEN_CHECK_S, UNSEEN_MARGIN_S,
)

logger = logging.getLogger(__name__)

__all__ = ["CameraManager", "_env_flag", "CoercibleAwaitable", "FramePacket", "FrameSlot", "pool_slots", "CameraWorkerState", "_TRUTHY", "_SHEDDING_STATES", "FRAME_POOL_BYTES", "BACKLOG_BYTES", "INSPECT_BATCH", "EVIDENCE_PENDING", "VERDICT_LOST_HOLD_S", "UNKNOWN_RECIPE", "UNSEEN_CHECK_S", "UNSEEN_MARGIN_S"]


class CameraManager(CaptureMixin, DefectMixin, InspectionMixin, VerdictMixin):
    """The station's cameras: starts and watches the workers, and owns what they share."""

    def __init__(self, config_path: str = "config.yaml", config_data: dict | None = None):
        self.config_path = config_path
        self.raw_config = self._load_raw_config(config_path, config_data)
        check_real_plc(self.raw_config, forced_by_env=_env_flag("QUALYXUS_USE_SYNTHETIC_CAMERAS")
                       or _env_flag("QUALYXUS_FORCE_MOCK_INFERENCE"))
        self.config = Config(raw=copy.deepcopy(self.raw_config))
        self.main_loop: asyncio.AbstractEventLoop | None = None

        self.shutdown_event = threading.Event()
        self._stopped = False
        # Part verdicts have their own lane: machine alerts can never crowd them out.
        self.fast_queue: queue.Queue = queue.Queue(maxsize=256)
        self.alert_queue: queue.Queue = queue.Queue(maxsize=20)
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
            require_ack=bool(telemetry_cfg.get("require_ack", False)),
            min_verdict_hold_s=float(telemetry_cfg.get("min_verdict_hold_s", 0.1)),
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
        self.stream_timeout_s = float(manager_cfg.get("stream_timeout_s", 5.0))
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
            max_gb=float(disk_cfg.get("max_gb", 20.0)),
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
        self.vlm_enabled = bool(vlm_cfg.get("enabled", False))
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
        self.presence: dict[str, PresenceDetector] = {
            str(c.get("id")): PresenceDetector.from_config({**(inspection_cfg.get("presence") or {}), **(c.get("presence") or {})})
            for c in (self.raw_config.get("cameras") or []) if c.get("enabled", True)}
        self._pending_parts: collections.deque = collections.deque()
        self._backlog_bytes = 0
        self._backlog_lock = threading.Lock()
        self.evidence_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="evidence-writer")
        self._evidence_pending = 0
        self._model_views = {str(c.get("id")): str(c["model_view"]) for c in (self.raw_config.get("cameras") or [])
                             if c.get("model_view")}
        trigger_cfg = inspection_cfg.get("trigger") or {}
        self.trigger: PartTrigger | None = None
        self.trigger_source: TriggerSource | None = None
        if trigger_cfg.get("enabled"):
            cameras = [str(c["id"]) for c in (self.raw_config.get("cameras") or []) if c.get("enabled", True)]
            self.trigger = PartTrigger(cameras, frames=int(trigger_cfg.get("frames", 3)),
                                       delay_s=float(trigger_cfg.get("delay_s", 0.05)),
                                       timeout_s=float(inspection_cfg.get("view_timeout_s", 5.0)))
            self.trigger_source = TriggerSource(self.telemetry.adapter.config, self._on_trigger,
                                                poll_s=float(trigger_cfg.get("poll_s", 0.02)))
        self._trigger_numbers: collections.OrderedDict[str, int] = collections.OrderedDict()
        self._presence_on_frames = int(inspection_cfg.get("on_frames", 2))
        self._presence_off_frames = int(inspection_cfg.get("off_frames", 2))
        self.part_trackers: dict[str, PartPresenceTracker] = {}
        self.inspection_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="part-inspection")
        self.part_fuser = PartVerdictFuser(
            required_views(self.raw_config.get("cameras") or []),
            timeout_s=float((self.raw_config.get("inspection") or {}).get("view_timeout_s", 5.0)),
        )
        self.heartbeat_s = float((self.raw_config.get("telemetry") or {}).get("heartbeat_s", 1.0))
        self._last_heartbeat = 0.0
        self.max_verdict_age_s = float((self.raw_config.get("telemetry") or {}).get("max_verdict_age_s", 3.0))
        self._alert_write = None
        self._reflex_stop = threading.Event()
        self.max_frame_age_s = float((self.raw_config.get("inspection") or {}).get("max_frame_age_s", 5.0))
        self._inspection_faulted = False
        self._view_faults: dict[str, bool] = {}
        self._delivery_fault_until = 0.0
        self._heartbeat_write = None
        self._hold_reason: str | None = None
        self._last_verdict: dict[str, Any] | None = None
        views_cfg = (self.raw_config.get("inspection") or {}).get("view_store") or {}
        self.view_store = (ViewStore(views_cfg.get("dir", "data/live/parts"), int(views_cfg.get("keep_parts", 200)))
                           if views_cfg.get("enabled", True) else None)
        self.frame_detector_missing = bool(getattr(self.multiplexer, "load_failed", False))
        inspect_cfg = self.raw_config.get("inspection") or {}
        self.max_inspect_s = float(inspect_cfg.get("max_inspect_s", 10.0))
        self._inspection_slots = threading.Semaphore(int(inspect_cfg.get("max_inspection_backlog", 64)))
        self._inspecting_since: float | None = None
        self.expect_part_per_cycle = bool(inspect_cfg.get("expect_part_per_cycle", True))
        self._cycles_checked_to: dict[str, float] = {}
        self._next_unseen_check = 0.0
        # Models per recipe (registry-versioned with shadow + hot reload, or explicit config paths).
        from src.inference.inspection_stack import StackProvider

        self.stack_provider = StackProvider(inspection_cfg)
        self.camera_recipes = {str(c.get("id")): c.get("recipe") for c in (self.raw_config.get("cameras") or [])
                               if c.get("id") is not None and c.get("recipe")}
        self.recipe_from_scalar = inspection_cfg.get("recipe_from_scalar")
        recipes = {self._model_recipe(str(c["id"]), self.camera_recipes.get(str(c["id"])) or self.stack_provider.default_recipe)
                   for c in (self.raw_config.get("cameras") or []) if c.get("enabled", True) and c.get("id") is not None}
        # Never fewer than the station needs at once, or every part would reload the models.
        self.stack_provider.max_cached = max(self.stack_provider.max_cached, 2 * len(recipes))
        for recipe in sorted(recipes) or [self.stack_provider.default_recipe]:
            self.stack_provider.get(recipe)  # at start-up, not on the first part
        self._health_cfg = inspection_cfg.get("health", {}) or {}
        self.model_health: dict[Any, Any] = {}
        self.retrain_on_drift = bool(self._health_cfg.get("request_retrain", True))
        review_cfg = inspection_cfg.get("review", {}) or {}
        self.review_queue = None
        if review_cfg.get("enabled", True):
            from src.inference.anomaly.monitoring import ReviewQueue

            self.review_queue = ReviewQueue(review_cfg.get("dir", "data/review"),
                                            band=tuple(review_cfg.get("band", (0.8, 1.25))),
                                            max_gb=float(review_cfg.get("max_gb", 5.0)))
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

    def attach_process_service(self, service) -> None:
        """Link inspections to machine cycles; render simulated parts for synthetic scenes."""
        self.process_service = service
        sources = {s.simulator.machine_id: s for s in getattr(service, "sources", [])
                   if hasattr(s, "outcomes_ended_between")}
        if not sources:
            return
        from src.process.synthetic_parts import SyntheticPartFeed

        for cam in self.raw_config.get("cameras", []) or []:
            if str(cam.get("synthetic_scene", "")).lower() != "process":
                continue
            link = service.aligner.link_for(str(cam.get("id")))
            delay = link.transport_delay_s if link else 4.0
            source = sources.get(link.machine_id) if link else next(iter(sources.values()))
            if source is None:
                continue
            self._part_feeds[str(cam.get("id"))] = SyntheticPartFeed(
                source, transport_delay_s=delay, dwell_s=float(cam.get("part_dwell_s", 0.6)),
                view_index=int(cam.get("view_index", 0)), n_views=int(cam.get("n_views", 1)),
                inject_defect_every=int(cam.get("inject_defect_every", 0)),
            )

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
        ids = [str(c.get("id")) for c in cameras]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate camera ids: {sorted(ids)}")
        return [c for c in cameras if c.get("enabled", True)]

    def _make_session_id(self, camera_id: str) -> str:
        return f"{camera_id}-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    def _start_camera_worker(self, camera_cfg: dict, previous_restarts: int = 0):
        camera_id = str(camera_cfg.get("id", "camera"))

        use_synth_env = _env_flag("QUALYXUS_USE_SYNTHETIC_CAMERAS")
        force_mock_env = _env_flag("QUALYXUS_FORCE_MOCK_INFERENCE")
        camera_cfg = dict(camera_cfg or {})
        if use_synth_env:
            camera_cfg["source"] = f"synthetic://{camera_id}"
            camera_cfg["mock_inference"] = True
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
                fast_queue=self.alert_queue,
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
        self._reflex_stop.clear()
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
            logger.debug("Ignored", exc_info=True)

        for cam in self._camera_entries():
            self._start_camera_worker(cam)
        if self.trigger_source is not None:
            self.trigger_source.start()

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

        # 3) Every part already seen gets its verdict, and the PLC gets it, before anything closes.
        self._finish_inspections(states)
        self._reflex_stop.set()
        for t in (self._reflex_thread, self._disk_writer_thread):
            if t is not None and t.is_alive():
                t.join(timeout=5.0)

        self.executor.shutdown(wait=True, cancel_futures=True)
        self.vlm_executor.shutdown(wait=True, cancel_futures=True)
        self.plc_executor.shutdown(wait=True)
        if getattr(self, "evidence_executor", None) is not None:
            self.evidence_executor.shutdown(wait=True)
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
        adapter = getattr(self.telemetry, "adapter", None)
        self._atomic_write_json(self.status_path, {
            "timestamp": time.time(), "workers": self.get_status_snapshot(),
            "problems": self.health_problems(),
            "plc": {"simulated": bool(getattr(adapter, "is_simulated", True)), "hold_reason": self._hold_reason,
                    "write_failures": int(getattr(adapter, "write_failures", 0)), "last_verdict": self._last_verdict},
        })
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

                    # Two threads must never read one camera: wait for the old ones to be gone. A stream read
                    # gives up after stream_timeout_s; until then the camera stays failed (heartbeat held).
                    for thread in (state.capture_thread, state.inference_thread):
                        if thread is not None and thread.is_alive():
                            thread.join(timeout=0.5)
                    if any(t is not None and t.is_alive() for t in (state.capture_thread, state.inference_thread)):
                        logger.error("[%s] The old capture has not ended yet; restart postponed.", cam_id)
                        continue
                    self._drop_visit(state)

                    cam_cfg = next((c for c in self._camera_entries() if str(c.get("id")) == cam_id), None)
                    if cam_cfg and not self.shutdown_event.is_set():
                        logger.info("[CameraManager] Respawning worker node %s (Attempt %d)...", cam_id, state.restarts + 1)
                        self._start_camera_worker(cam_cfg, previous_restarts=state.restarts + 1)

                elif state.frames_captured > 10 and not state.failed:
                    self.watchdog.record_recovered(cam_id)

            if not self.shutdown_event.is_set():
                self._publish_runtime_status()


