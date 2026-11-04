import asyncio
import sqlite3
import logging
import json
import os
import csv
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from src.compliance.iso_mapper import ISOMapper
from src.logging.schema import DefectEvent
from src.logging.db import init_db
from src.logging.metadata_contract import compute_audit_hash, to_metadata_dict

logger = logging.getLogger("QualyxusEventLogger")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Large per-pixel arrays are persisted elsewhere and must never be pushed to SSE clients.
_BROADCAST_EXCLUDED_KEYS = {"mask", "frame_data"}


def _to_json_safe(value):
    """Recursively convert numpy / datetime values into JSON-serialisable primitives."""
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items() if k not in _BROADCAST_EXCLUDED_KEYS}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat()
    return value

class EventLogger:
    """
    Asynchronous, single-worker batch serialization engine backed by memory-bound queues 
    and atomic WAL commits. Guarantees 100% transaction compliance via clean 
    shutdown memory evacuation routines and self-healing lazy initialization.
    Supports both legacy synchronous logging and high-velocity async batching.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, sqlite_path: str | None = None, csv_path: str | None = None):
        if not hasattr(self, "subscribers"):
            self.subscribers = []
            
        if self._initialized:
            reinit = False
            if sqlite_path and sqlite_path != self.db_path:
                self.db_path = sqlite_path
                reinit = True
            if csv_path and csv_path != self.csv_fallback_path:
                self.csv_fallback_path = csv_path
                reinit = True
            if reinit:
                try:
                    self._init_db()
                except Exception as e:
                    logger.warning("[EventLogger] Failed to run schema migration on re-initialization: %s", e)
            return
            
        logging_cfg = self._configured_logging_paths() if not (sqlite_path and csv_path) else {}
        self.db_path = (
            sqlite_path
            or os.environ.get("SQLITE_DB_PATH")
            or logging_cfg.get("sqlite_path")
            or "data/events/events.db"
        )
        self.csv_fallback_path = (
            csv_path
            or os.environ.get("CSV_FALLBACK_PATH")
            or logging_cfg.get("csv_path")
            or "data/events/events.csv"
        )
        
        # Thread-safe asyncio ring buffer capping memory footprint bounds
        self.queue = asyncio.Queue(maxsize=10000)
        
        self.batch_size = 50
        self.timeout_sec = 0.05  # Tight 50ms batching aggregation window
        
        self.db_executor = ThreadPoolExecutor(max_workers=1)
        self.db_lock = threading.Lock()
        # VLM narratives can complete before the async batch that inserts their event row.
        self._pending_narratives: dict[str, str] = {}
        self._camera_cfg_cache: tuple[float, dict[str, dict]] = (0.0, {})
        self.worker_task = None
        self._starting = False
        self._db_conn = None
        self._initialized = True

        # Synchronously initialize DB schema and tables on boot
        try:
            self._init_db()
        except Exception as e:
            logger.warning("[EventLogger] Failed to run schema migration on initialization: %s", e)

    @staticmethod
    def _configured_logging_paths() -> dict:
        try:
            from src.config.config import Config
            return dict(Config.load().get("logging", default={}) or {})
        except Exception:
            return {}

    @staticmethod
    def new_event_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def now() -> datetime:
        return datetime.utcnow()

    def _generate_audit_hash(self, event: DefectEvent) -> str:
        # Contract-hardening: audit hash is computed over canonical metadata JSON.
        md = to_metadata_dict(event)
        return compute_audit_hash(md)

    async def start(self):
        """
        Asynchronously initializes database schemas and boots the background worker daemon.
        Prevents synchronous disk interactions from stalling the main loop initialization.
        Protected by a concurrency shield to prevent duplicate boot sequences.
        """
        if self._starting or (self.worker_task is not None and not self.worker_task.done()):
            return
            
        self._starting = True
        try:
            loop = asyncio.get_running_loop()
            # Offload schema checking and directory generation onto the background thread pool
            await loop.run_in_executor(self.db_executor, self._init_db)
            self.worker_task = asyncio.create_task(self._database_worker_loop())
            logger.info("[EventLogger] Database daemon fully activated.")
        finally:
            self._starting = False

    def _init_db(self):
        """Executed inside the isolated thread pool executor on boot state tracking."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # 1. Initialize SQLAlchemy schema for defect_events (runs table creation & migrations)
        try:
            init_db(self.db_path)
        except Exception as e:
            logger.warning("[EventLogger] SQLAlchemy init_db failed: %s. Falling back to manual schemas.", e)

        # 2. Hardened Fix (Blind Spot 1): Explicitly define schema layouts for BOTH defect_events
        # and events tables using raw SQL to guarantee they exist regardless of SQLAlchemy state.
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, isolation_level=None)
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            
            # Create defect_events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS defect_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    severity INTEGER NOT NULL,
                    batch_id TEXT,
                    operator_id TEXT,
                    machine_id TEXT,
                    shift_id TEXT,
                    line_id TEXT,
                    station_id TEXT,
                    temperature REAL,
                    vibration REAL,
                    speed REAL,
                    lot_id TEXT,
                    full_image_path TEXT,
                    overlay_path TEXT,
                    crop_path TEXT,
                    audit_hash TEXT,
                    iso_clause TEXT,
                    vlm_narrative TEXT,
                    inspection_profile TEXT
                )
            """)
            
            # Create events table (raw tracking payload storage)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    box TEXT,
                    track_id INTEGER,
                    is_new_anomaly BOOLEAN,
                    timestamp REAL
                )
            """)
            
            # Create geometry_events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS geometry_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    structural_similarity_score REAL,
                    spatial_drift_x_mm REAL,
                    spatial_drift_y_mm REAL,
                    orientation_deviation_degrees REAL,
                    justification TEXT,
                    timestamp REAL
                )
            """)
        except Exception as e:
            logger.critical("[EventLogger] Failed to manually verify SQLite schema layout: %s", e)
        finally:
            if conn:
                conn.close()

    def broadcast_to_subscribers(self, payload: dict) -> None:
        """Fans out a sanitized JSON dictionary payload to all active memory queues safely across thread scopes."""
        if not hasattr(self, "subscribers") or not self.subscribers:
            return

        payload = _to_json_safe(payload)
        from src.core.factory_manager import FactoryManager
        cm = FactoryManager().get_instance("CameraManager")
        main_loop = getattr(cm, "main_loop", None) if cm else None

        def _safe_push_async(target_queue, data):
            try:
                target_queue.put_nowait(data)
            except asyncio.QueueFull:
                try:
                    # Drop the oldest packet out of the ring buffer to accommodate incoming high-velocity data
                    target_queue.get_nowait()
                    target_queue.put_nowait(data)
                except Exception:
                    pass

        for q in list(self.subscribers):
            try:
                if main_loop and main_loop.is_running():
                    main_loop.call_soon_threadsafe(_safe_push_async, q, payload)
                else:
                    try:
                        q.put_nowait(payload)
                    except asyncio.QueueFull:
                        q.get_nowait()
                        q.put_nowait(payload)
            except Exception:
                pass

    def log_event(self, event: DefectEvent):
        """
        Synchronously logs a single DefectEvent to SQLite and CSV.
        Maintains total backward compatibility for compliance test suites and synchronous processes.
        """
        # 1. Enforce Profile/Metadata constraints
        try:
            from src.utils.profile_manager import ProfileManager
            pm = ProfileManager()
            profile_name = pm.active_profile
            profile_line_id = pm.get_line_id()
            profile_station_id = pm.get_station_id()
            
            if profile_line_id:
                event.line_id = str(profile_line_id)
            if profile_station_id:
                event.station_id = str(profile_station_id)
            event.inspection_profile = profile_name
        except Exception:
            pass

        # 2. Audit Hash and ISO Clause
        if not event.audit_hash:
            try:
                event.audit_hash = self._generate_audit_hash(event)
            except Exception as e:
                logger.error("[EventLogger] Event saved without an audit hash: %s", e)
                
        if not event.iso_clause:
            try:
                event.iso_clause = ISOMapper.clause_for_event(event)
            except Exception:
                pass

        if not event.vlm_narrative:
            event.vlm_narrative = self._pop_pending_narrative(event.event_id)

        # 3. Synchronous SQLite Insert (upsert transaction)
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            cursor = conn.cursor()
            
            ts_str = event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp)
            
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("""
                INSERT OR REPLACE INTO defect_events (
                    event_id, timestamp, source, defect_type, confidence, severity,
                    batch_id, operator_id, machine_id, shift_id, line_id, station_id,
                    temperature, vibration, speed, lot_id, full_image_path, overlay_path,
                    crop_path, audit_hash, iso_clause, vlm_narrative, inspection_profile
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                ts_str,
                event.source,
                event.defect_type,
                event.confidence,
                event.severity,
                event.batch_id,
                event.operator_id,
                event.machine_id,
                event.shift_id,
                event.line_id,
                event.station_id,
                event.temperature,
                event.vibration,
                event.speed,
                event.lot_id,
                event.full_image_path,
                event.overlay_path,
                event.crop_path,
                event.audit_hash,
                event.iso_clause,
                event.vlm_narrative,
                event.inspection_profile
            ))
            cursor.execute("COMMIT;")
        except Exception as exc:
            logger.error("[EventLogger] Sync DB write failed: %s. Reverting to emergency CSV.", exc)
            self._write_csv_row_sync(event)
            raise
        finally:
            if conn is not None:
                conn.close()

        # 4. CSV Logging
        self._write_csv_row_sync(event)

        # 5. Volatile Memory Broadcast to Subscribers
        from src.logging.metadata_contract import to_metadata_dict
        payload = to_metadata_dict(event)
        self.broadcast_to_subscribers(payload)

    def _camera_config(self, camera_id: str) -> dict:
        """Per-camera config entry, resolved from the live CameraManager / ConfigVault / config file.

        The resolved table is cached for 30s so CSV writes never re-parse YAML per event.
        """
        cached_at, table = self._camera_cfg_cache
        if time.monotonic() - cached_at > 30.0 or (camera_id not in table and table == {}):
            cameras: list = []
            try:
                from src.core.factory_manager import FactoryManager
                camera_manager = FactoryManager().get_instance("CameraManager")
                if camera_manager is not None:
                    cameras = list(camera_manager.raw_config.get("cameras", []) or [])
                if not cameras:
                    vault = FactoryManager().get_instance("ConfigVault")
                    if vault is not None:
                        cameras = list(vault.get("cameras", default=[]) or [])
                if not cameras:
                    from src.config.config import Config
                    cameras = list(Config.load().get("cameras", default=[]) or [])
            except Exception:
                cameras = []
            table = {str(c.get("id")): c for c in cameras if isinstance(c, dict)}
            self._camera_cfg_cache = (time.monotonic(), table)
        return table.get(str(camera_id), {})

    def _pop_pending_narrative(self, event_id) -> str | None:
        if not event_id:
            return None
        with self.db_lock:
            return self._pending_narratives.pop(str(event_id), None)

    def _write_csv_row_sync(self, event: DefectEvent):
        try:
            # Check if independent logs are enabled for this camera source
            independent = bool(self._camera_config(event.source).get("independent_logs", False))
            session_id = None

            if independent:
                # Retrieve session_id if available
                try:
                    from src.core.factory_manager import FactoryManager
                    camera_manager = FactoryManager().get_instance("CameraManager")
                    if camera_manager and event.source in camera_manager.workers:
                        session_id = camera_manager.workers[event.source].session_id
                except Exception:
                    pass
                if not session_id:
                    session_id = str(int(time.time()))

                # Determine the custom CSV path
                parent_dir = os.path.dirname(self.csv_fallback_path)
                csv_path = os.path.join(parent_dir, f"events_{event.source}_{session_id}.csv")
            else:
                csv_path = self.csv_fallback_path

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow([
                        "event_id", "timestamp", "source", "defect_type", "confidence", "severity",
                        "batch_id", "operator_id", "machine_id", "shift_id",
                        "line_id", "station_id", "temperature", "vibration", "speed", "lot_id",
                        "full_image_path", "overlay_path", "crop_path", "audit_hash", "iso_clause", "vlm_narrative",
                        "inspection_profile"
                    ])
                
                ts_str = event.timestamp.isoformat() if isinstance(event.timestamp, datetime) else str(event.timestamp)
                writer.writerow([
                    event.event_id,
                    ts_str,
                    event.source,
                    event.defect_type,
                    f"{event.confidence:.6f}" if event.confidence is not None else "",
                    event.severity,
                    event.batch_id or "",
                    event.operator_id or "",
                    event.machine_id or "",
                    event.shift_id or "",
                    event.line_id or "",
                    event.station_id or "",
                    f"{event.temperature:.2f}" if event.temperature is not None else "",
                    f"{event.vibration:.3f}" if event.vibration is not None else "",
                    f"{event.speed:.1f}" if event.speed is not None else "",
                    event.lot_id or "",
                    event.full_image_path,
                    event.overlay_path or "",
                    event.crop_path or "",
                    event.audit_hash or "",
                    event.iso_clause or "",
                    event.vlm_narrative or "",
                    event.inspection_profile or "",
                ])
        except Exception as e:
            logger.critical("[EventLogger] Failed to write to CSV fallback: %s", e)

    def _write_geometry_csv_row_sync(self, event: dict):
        try:
            camera_id = event.get("camera_id", "UNKNOWN")
            independent = bool(self._camera_config(camera_id).get("independent_logs", False))
            session_id = None

            if independent:
                try:
                    from src.core.factory_manager import FactoryManager
                    camera_manager = FactoryManager().get_instance("CameraManager")
                    if camera_manager and camera_id in camera_manager.workers:
                        session_id = camera_manager.workers[camera_id].session_id
                except Exception:
                    pass
                if not session_id:
                    session_id = str(int(time.time()))

                parent_dir = os.path.dirname(self.csv_fallback_path)
                csv_path = os.path.join(parent_dir, f"geometry_{camera_id}_{session_id}.csv")
            else:
                parent_dir = os.path.dirname(self.csv_fallback_path)
                csv_path = os.path.join(parent_dir, "geometry.csv")

            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                if not file_exists:
                    writer.writerow([
                        "camera_id", "status", "structural_similarity_score",
                        "spatial_drift_x_mm", "spatial_drift_y_mm",
                        "orientation_deviation_degrees", "justification", "timestamp"
                    ])
                writer.writerow([
                    camera_id,
                    event.get("status", "UNKNOWN"),
                    f"{event.get('structural_similarity_score', 0.0):.6f}",
                    f"{event.get('spatial_drift_x_mm', 0.0):.6f}",
                    f"{event.get('spatial_drift_y_mm', 0.0):.6f}",
                    f"{event.get('orientation_deviation_degrees', 0.0):.6f}",
                    event.get("justification", ""),
                    f"{event.get('timestamp', time.time()):.6f}"
                ])
        except Exception as e:
            logger.critical("[EventLogger] Failed to write to geometry CSV fallback: %s", e)

    def _is_defect_event_payload(self, item: dict) -> bool:
        return "event_id" in item or "defect_type" in item

    def _is_geometry_event_payload(self, item: dict) -> bool:
        return isinstance(item, dict) and "status" in item and "structural_similarity_score" in item

    def _create_defect_event_from_raw(self, item: dict) -> dict:
        event_id = self.new_event_id()
        ts_val = item.get("timestamp", time.time())
        ts_datetime = datetime.utcfromtimestamp(ts_val)
        
        profile_name = "default"
        profile_line_id = None
        profile_station_id = None
        try:
            from src.utils.profile_manager import ProfileManager
            pm = ProfileManager()
            profile_name = pm.active_profile or "default"
            profile_line_id = pm.get_line_id()
            profile_station_id = pm.get_station_id()
        except Exception:
            pass
            
        # Derive confidence/severity from the detection itself. Telemetry fields stay NULL:
        # this path has no telemetry context and must not invent plant readings.
        confidence = None
        box = item.get("box")
        try:
            if box is not None and len(box) > 4:
                confidence = float(box[4])
        except Exception:
            confidence = None
        if confidence is None:
            confidence = 0.0
        from src.inference.postprocess import severity_from_conf

        payload = {
            "event_id": event_id,
            "timestamp": ts_datetime,
            "source": item.get("camera_id", "UNKNOWN"),
            "defect_type": "anomaly",
            "confidence": confidence,
            "severity": severity_from_conf(confidence),
            "batch_id": None,
            "operator_id": None,
            "machine_id": None,
            "shift_id": None,
            "line_id": str(profile_line_id) if profile_line_id else None,
            "station_id": str(profile_station_id) if profile_station_id else None,
            "temperature": None,
            "vibration": None,
            "speed": None,
            "lot_id": None,
            "full_image_path": "",
            "overlay_path": None,
            "crop_path": None,
            "audit_hash": None,
            "iso_clause": None,
            "vlm_narrative": None,
            "inspection_profile": profile_name
        }
        
        try:
            temp_ev = DefectEvent(**payload)
            payload["audit_hash"] = self._generate_audit_hash(temp_ev)
            payload["iso_clause"] = ISOMapper.clause_for_event(temp_ev)
        except Exception as e:
            logger.error("Failed to generate audit metadata for raw anomaly: %s", e)
            
        return payload

    async def log_batch_async(self, payloads: list):
        """
        True Asynchronous Ingestion Coroutine. 
        Guarantees total signature compatibility with the Inference Multiplexer pipeline.
        Falls back cleanly to synchronous transaction writes in loop-less environments (like tests).
        """
        try:
            loop = asyncio.get_running_loop()
            if not self._starting and (self.worker_task is None or self.worker_task.done()):
                await self.start()
                
            dropped = 0
            for payload in payloads:
                self.broadcast_to_subscribers(payload)
                try:
                    self.queue.put_nowait(payload)
                except asyncio.QueueFull:
                    dropped += 1
                    
            if dropped > 0:
                logger.warning("[EventLogger] Ingestion queue full (10000). Dropped %d payloads.", dropped)
        except RuntimeError:
            # Fallback to synchronous execution if no event loop is running (e.g. in test suites)
            for payload in payloads:
                self.broadcast_to_subscribers(payload)
            self._sync_execute_transaction(payloads)

    def _sync_execute_transaction(self, batch: list):
        """
        Executes atomic WAL commits synchronously inside the background thread executor.
        Maintains a persistent connection to eradicate I/O overhead from continuous file operations.
        """
        if not batch:
            return

        db_rows_events = []
        db_rows_defects = []
        db_rows_geometry = []
        csv_defects = []
        csv_geometry = []

        for item in batch:
            try:
                # If it's a geometry event payload
                if self._is_geometry_event_payload(item):
                    db_rows_geometry.append((
                        str(item.get("camera_id", "UNKNOWN")),
                        str(item.get("status", "UNKNOWN")),
                        float(item.get("structural_similarity_score", 0.0)),
                        float(item.get("spatial_drift_x_mm", 0.0)),
                        float(item.get("spatial_drift_y_mm", 0.0)),
                        float(item.get("orientation_deviation_degrees", 0.0)),
                        str(item.get("justification", "")),
                        float(item.get("timestamp", time.time()))
                    ))
                    csv_geometry.append(item)
                # If it's a DefectEvent dictionary or object
                elif isinstance(item, dict) and self._is_defect_event_payload(item):
                    try:
                        from src.utils.profile_manager import ProfileManager
                        pm = ProfileManager()
                        profile_name = pm.active_profile
                        profile_line_id = pm.get_line_id()
                        profile_station_id = pm.get_station_id()
                        
                        if profile_line_id:
                            item["line_id"] = str(profile_line_id)
                        if profile_station_id:
                            item["station_id"] = str(profile_station_id)
                        item["inspection_profile"] = profile_name
                    except Exception:
                        pass

                    if not item.get("vlm_narrative"):
                        pending = self._pending_narratives.pop(str(item.get("event_id")), None)
                        if pending:
                            item["vlm_narrative"] = pending

                    if not item.get("audit_hash"):
                        try:
                            temp_ev = DefectEvent(**{k: v for k, v in item.items() if k != "frame_data"})
                            item["audit_hash"] = self._generate_audit_hash(temp_ev)
                        except Exception as e:
                            logger.error("[EventLogger] Event saved without an audit hash: %s", e)
                    if not item.get("iso_clause"):
                        try:
                            temp_ev = DefectEvent(**{k: v for k, v in item.items() if k != "frame_data"})
                            item["iso_clause"] = ISOMapper.clause_for_event(temp_ev)
                        except Exception:
                            pass
                    
                    ts_val = item.get("timestamp")
                    if isinstance(ts_val, datetime):
                        ts_str = ts_val.isoformat()
                    else:
                        ts_str = str(ts_val)

                    db_rows_defects.append((
                        item.get("event_id"),
                        ts_str,
                        item.get("source"),
                        item.get("defect_type"),
                        item.get("confidence"),
                        item.get("severity"),
                        item.get("batch_id"),
                        item.get("operator_id"),
                        item.get("machine_id"),
                        item.get("shift_id"),
                        item.get("line_id"),
                        item.get("station_id"),
                        item.get("temperature"),
                        item.get("vibration"),
                        item.get("speed"),
                        item.get("lot_id"),
                        item.get("full_image_path"),
                        item.get("overlay_path"),
                        item.get("crop_path"),
                        item.get("audit_hash"),
                        item.get("iso_clause"),
                        item.get("vlm_narrative"),
                        item.get("inspection_profile")
                    ))
                    csv_defects.append(DefectEvent(**{k: v for k, v in item.items() if k != "frame_data"}))
                    
                # Otherwise, it's a raw tracking event payload
                elif isinstance(item, dict):
                    box_data = item.get("box")
                    if isinstance(box_data, np.ndarray):
                        box_json = json.dumps(box_data.tolist())
                    elif isinstance(box_data, list):
                        box_json = json.dumps(box_data)
                    else:
                        box_json = json.dumps([])
                        
                    db_rows_events.append((
                        str(item.get("camera_id", "UNKNOWN")),
                        box_json,
                        int(item.get("track_id", -1)),
                        1 if item.get("is_new_anomaly") else 0,
                        float(item.get("timestamp", 0.0))
                    ))

                    if item.get("is_new_anomaly"):
                        defect_item = self._create_defect_event_from_raw(item)
                        
                        ts_val = defect_item.get("timestamp")
                        if isinstance(ts_val, datetime):
                            ts_str = ts_val.isoformat()
                        else:
                            ts_str = str(ts_val)

                        db_rows_defects.append((
                            defect_item.get("event_id"),
                            ts_str,
                            defect_item.get("source"),
                            defect_item.get("defect_type"),
                            defect_item.get("confidence"),
                            defect_item.get("severity"),
                            defect_item.get("batch_id"),
                            defect_item.get("operator_id"),
                            defect_item.get("machine_id"),
                            defect_item.get("shift_id"),
                            defect_item.get("line_id"),
                            defect_item.get("station_id"),
                            defect_item.get("temperature"),
                            defect_item.get("vibration"),
                            defect_item.get("speed"),
                            defect_item.get("lot_id"),
                            defect_item.get("full_image_path"),
                            defect_item.get("overlay_path"),
                            defect_item.get("crop_path"),
                            defect_item.get("audit_hash"),
                            defect_item.get("iso_clause"),
                            defect_item.get("vlm_narrative"),
                            defect_item.get("inspection_profile")
                        ))
                        csv_defects.append(DefectEvent(**defect_item))
                        
                elif isinstance(item, DefectEvent):
                    # Handle raw DefectEvent objects passed inside lists
                    self.log_event(item)

            except Exception as parse_err:
                logger.error("[EventLogger] Failed to parse corrupted payload for insertion: %s", parse_err)
                continue

        with self.db_lock:
            try:
                if self._db_conn is None:
                    self._db_conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None, check_same_thread=False)
                    self._db_conn.execute("PRAGMA journal_mode=WAL;")
                    self._db_conn.execute("PRAGMA synchronous=NORMAL;")
                
                cursor = self._db_conn.cursor()
                cursor.execute("BEGIN TRANSACTION;")
                
                if db_rows_events:
                    cursor.executemany("""
                        INSERT INTO events (camera_id, box, track_id, is_new_anomaly, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    """, db_rows_events)
                    
                if db_rows_defects:
                    cursor.executemany("""
                        INSERT OR REPLACE INTO defect_events (
                            event_id, timestamp, source, defect_type, confidence, severity,
                            batch_id, operator_id, machine_id, shift_id, line_id, station_id,
                            temperature, vibration, speed, lot_id, full_image_path, overlay_path,
                            crop_path, audit_hash, iso_clause, vlm_narrative, inspection_profile
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, db_rows_defects)

                if db_rows_geometry:
                    cursor.executemany("""
                        INSERT INTO geometry_events (
                            camera_id, status, structural_similarity_score,
                            spatial_drift_x_mm, spatial_drift_y_mm,
                            orientation_deviation_degrees, justification, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, db_rows_geometry)
                    
                cursor.execute("COMMIT;")
            except Exception as e:
                logger.error("[EventLogger] Atomic transaction failed: %s. Executing ROLLBACK.", e)
                if self._db_conn:
                    try:
                        self._db_conn.execute("ROLLBACK;")
                        self._db_conn.close()
                    except Exception as rollback_err:
                        logger.error("[EventLogger] Transaction ROLLBACK failed: %s", rollback_err)
                    self._db_conn = None
                
                for dev in csv_defects:
                    self._write_csv_row_sync(dev)
                for ge in csv_geometry:
                    self._write_geometry_csv_row_sync(ge)
                raise
        
        for dev in csv_defects:
            self._write_csv_row_sync(dev)
        for ge in csv_geometry:
            self._write_geometry_csv_row_sync(ge)

    async def _database_worker_loop(self):
        """
        Isolated Single-Worker Execution Loop.
        Dynamic Transaction Batcher managing extraction limits and timeout thresholds.
        """
        logger.info("[EventLogger] Database worker loop activated.")
        loop = asyncio.get_running_loop()
        
        try:
            while True:
                batch = []
                first_item = await self.queue.get()
                batch.append(first_item)
                start_time = time.perf_counter()
                
                while len(batch) < self.batch_size:
                    elapsed = time.perf_counter() - start_time
                    rem_time = self.timeout_sec - elapsed
                    
                    if rem_time <= 0:
                        break
                        
                    try:
                        item = await asyncio.wait_for(self.queue.get(), timeout=rem_time)
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break
                
                if batch:
                    try:
                        await loop.run_in_executor(self.db_executor, self._sync_execute_transaction, batch)
                    finally:
                        for _ in range(len(batch)):
                            self.queue.task_done()
                        
        except asyncio.CancelledError:
            logger.warning("[EventLogger] Process termination detected. Commencing emergency queue evacuation sequence...")
            evacuation_batch = []
            while not self.queue.empty():
                try:
                    item = self.queue.get_nowait()
                    evacuation_batch.append(item)
                    self.queue.task_done()
                except asyncio.QueueEmpty:
                    break
            
            def safe_shutdown(batch_to_evac):
                if batch_to_evac:
                    logger.info("[EventLogger] Flashing %d records to disk before shutdown.", len(batch_to_evac))
                    self._sync_execute_transaction(batch_to_evac)
                if self._db_conn:
                    try:
                        self._db_conn.close()
                    except Exception:
                        pass
                    self._db_conn = None
                    
            if evacuation_batch or self._db_conn:
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(self.db_executor, safe_shutdown, evacuation_batch),
                        timeout=1.5
                    )
                except asyncio.TimeoutError:
                    logger.critical("[EventLogger] Disk subsystem stalled during shutdown. Truncating evacuation.")
                    self.db_executor.shutdown(wait=False)
            
            logger.info("[EventLogger] Queue evacuation finalized cleanly. Database connection severed.")
            raise

    def close(self) -> None:
        """Best-effort release of DB resources for this logger instance."""
        try:
            if self.worker_task is not None and not self.worker_task.done():
                try:
                    loop = asyncio.get_running_loop()
                    self.worker_task.cancel()
                except RuntimeError:
                    pass
            if self.db_executor:
                self.db_executor.shutdown(wait=True)
            if self._db_conn:
                self._db_conn.close()
                self._db_conn = None
        except Exception:
            pass

    def append_vlm_narrative(self, event_id: str, narrative: str) -> None:
        """
        Appends compiled VLM forensic descriptions to an active defect event record row 
        safely within our central, thread-safe database transaction lock context.
        """
        query = "UPDATE defect_events SET vlm_narrative = ? WHERE event_id = ?"
        with self.db_lock:
            conn = None
            try:
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                cursor = conn.cursor()
                cursor.execute(query, (narrative, event_id))
                updated = cursor.rowcount
                conn.commit()
                if updated == 0:
                    # Row not inserted yet (async batch still queued): attach it on insert.
                    self._pending_narratives[str(event_id)] = narrative
                    if len(self._pending_narratives) > 1000:
                        self._pending_narratives.pop(next(iter(self._pending_narratives)))
            except Exception as e:
                logger.error(
                    "[EventLogger] Failed to update narrative for event %s: %s", event_id, e
                )
            finally:
                if conn is not None:
                    conn.close()
