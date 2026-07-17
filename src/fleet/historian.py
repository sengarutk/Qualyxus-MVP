import asyncio
import logging
import os
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.core.factory_manager import FactoryManager

logger = logging.getLogger("QualyxusHistorian")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class PredictiveHistorian:
    """
    Non-blocking Time-Series Database Architect & RUL Computation Engine.
    Persists tracking matrices and vibration telemetry into a WAL-optimized
    SQLite transactional store without stalling the main async event loop.
    """
    def __init__(self, db_path: str = "data/historian/telemetry.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Dedicated execution boundary for isolated disk IO to keep event loop un-throttled
        self.io_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="historian_io")
        self._lock = threading.Lock()
        
        # RUL Bounded Cache (Sliding window of the last 1000 snapshots)
        self.max_cache_size = 1000
        self.snapshot_history = deque(maxlen=self.max_cache_size)
        
        # ISO 10816-3 Critical Zone D Boundary (mm/s)
        self.critical_vibration_threshold = 11.2
        
        self._trend_slope = 0.0
        self._rul_days = -1.0
        
        self._initialize_database()
        
        # Deterministic Singleton Mapping
        FactoryManager().register_instance("PredictiveHistorian", self)
        logger.info("[PredictiveHistorian] Instantiated and wired into FactoryManager registry.")

    def _get_connection(self) -> sqlite3.Connection:
        """Centralized connection context factory enforcing per-connection safety PRAGMAs."""
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        cursor = conn.cursor()
        # Enforce strict production durability and execution constraints per handle context
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _initialize_database(self):
        """Bootstraps the SQLite storage schema and persistent parameters."""
        try:
            with sqlite3.connect(self.db_path, timeout=5.0) as conn:
                cursor = conn.cursor()
                # WAL mode stays persistent within the database file header itself
                cursor.execute("PRAGMA journal_mode=WAL;")
                
                # Tracking payload storage
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS tracking_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        camera_id TEXT,
                        track_id INTEGER,
                        is_anomaly INTEGER,
                        bbox TEXT
                    )
                ''')
                
                # Vibration baseline storage
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS vibration_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        rms REAL,
                        dominant_freq REAL,
                        iso_status TEXT
                    )
                ''')
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracking_ts ON tracking_history(timestamp);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_vib_ts ON vibration_snapshots(timestamp);")
                conn.commit()
        except sqlite3.Error as e:
            logger.error("[PredictiveHistorian] Failed to bootstrap database schema: %s", e)

    def _sync_write_payloads(self, payloads: list):
        """Isolated transactional thread worker for inserting multiplexer payloads."""
        if not payloads:
            return
            
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                data_tuples = [
                    (
                        p.get("timestamp", time.time()),
                        p.get("camera_id", "unknown"),
                        p.get("track_id", -1),
                        1 if p.get("is_new_anomaly", False) else 0,
                        str(p.get("box", []))
                    )
                    for p in payloads
                ]
                cursor.executemany('''
                    INSERT INTO tracking_history (timestamp, camera_id, track_id, is_anomaly, bbox)
                    VALUES (?, ?, ?, ?, ?)
                ''', data_tuples)
                conn.commit()
        except sqlite3.Error as e:
            logger.error("[PredictiveHistorian] Transactional write failed: %s", e)

    async def process_history_async(self, detached_payloads: list):
        """Asynchronously ingests tracking matrices via an isolated executor block."""
        if not detached_payloads:
            return
            
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self.io_executor, self._sync_write_payloads, detached_payloads)

    def _sync_write_snapshot(self, timestamp: float, rms: float, dominant_freq: float, iso_status: str):
        """Writes vibration snapshots to disk atomically utilizing connection factory properties."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO vibration_snapshots (timestamp, rms, dominant_freq, iso_status)
                    VALUES (?, ?, ?, ?)
                ''', (timestamp, rms, dominant_freq, iso_status))
                conn.commit()
        except sqlite3.Error as e:
            logger.error("[PredictiveHistorian] Failed to commit snapshot: %s", e)

    def record_snapshot(self, rms: float, dominant_freq: float, iso_status: str):
        """Ingests vibration metrics, offloads disk IO, and updates sliding regression trend vectors."""
        current_ts = time.time()
        
        # Enqueue non-blocking DB transaction context
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(self.io_executor, self._sync_write_snapshot, current_ts, rms, dominant_freq, iso_status)
        except RuntimeError:
            # Fallback worker execution block if the calling thread has no active running loop context
            self.io_executor.submit(self._sync_write_snapshot, current_ts, rms, dominant_freq, iso_status)

        # Update analytical degradation matrix
        with self._lock:
            self.snapshot_history.append((current_ts, rms))
            
            if len(self.snapshot_history) > 10:
                try:
                    data = np.array(self.snapshot_history)
                    times = data[:, 0]
                    rmss = data[:, 1]
                    
                    # Hardening Guard: Ensure variance exists in the timing array to protect against singular matrix crashes
                    if np.max(times) - np.min(times) > 1e-5:
                        # Normalize time values to days to prevent floating-point catastrophic cancellation faults
                        time_days = (times - times[0]) / 86400.0
                        
                        # Compute degradation derivative (slope: mm/s per day)
                        slope, intercept = np.polyfit(time_days, rmss, 1)
                        self._trend_slope = float(slope)
                        
                        # RUL Projection logic
                        if self._trend_slope > 1e-6:
                            current_rms = rmss[-1]
                            remaining_rms = self.critical_vibration_threshold - current_rms
                            if remaining_rms > 0:
                                self._rul_days = remaining_rms / self._trend_slope
                            else:
                                self._rul_days = 0.0
                        else:
                            self._rul_days = -1.0  # Baseline is perfectly stable or actively improving
                except (np.linalg.LinAlgError, ValueError):
                    # Graceful recovery: maintain previous historical values if matrix calculations fault
                    pass

    @property
    def current_trend_slope(self) -> float:
        """Returns the continuous linear regression degradation slope (units/day)."""
        with self._lock:
            return self._trend_slope

    @property
    def current_rul_days(self) -> float:
        """Returns the estimated Remaining Useful Life projection in days."""
        with self._lock:
            return self._rul_days

    def close(self):
        """Secure file header closure and thread pool eviction during system-wide shutdowns."""
        logger.warning("[PredictiveHistorian] Executing deterministic thread shutdown and draining DB write queue.")
        # Block and force thread executor to flush remaining entries completely to disk before exit
        self.io_executor.shutdown(wait=True)
        logger.info("[PredictiveHistorian] Storage pool closed cleanly.")
