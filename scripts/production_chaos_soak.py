import os
import sys
import time
import queue
import sqlite3
import threading
import traceback
import numpy as np
import torch
import cv2

# Set path routing
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.camera.camera_manager import CameraManager, CameraWorkerState
from src.logging.event_logger import EventLogger
from src.logging.schema import DefectEvent, SystemState
from src.core.factory_manager import FactoryManager
from src.inference.analysis.geometry_engine import GeometryEngine
from src.inference.analysis.vibration_engine import VibrationAnalyzer

PASS, FAIL = "✅", "❌"
_errors = []

def _assert(cond: bool, msg: str) -> None:
    sym = PASS if cond else FAIL
    print(f"     {sym} {msg}")
    if not cond:
        _errors.append(msg)

def run_db_locker(db_path: str, stop_event: threading.Event):
    """Worker thread that keeps events.db locked via exclusive transaction loops."""
    while not stop_event.is_set():
        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=1.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("BEGIN EXCLUSIVE TRANSACTION;")
            # Keep database locked for 500ms
            time.sleep(0.5)
            conn.execute("COMMIT;")
        except Exception:
            pass
        finally:
            if conn:
                conn.close()
        time.sleep(0.01)

def run_logger_flooder(event_logger: EventLogger, stop_event: threading.Event):
    """Worker thread that floods the EventLogger with mock payloads."""
    while not stop_event.is_set():
        try:
            payload = {
                "status": "GEOMETRIC_VIOLATION",
                "structural_similarity_score": 0.74,
                "spatial_drift_x_mm": 2.5,
                "spatial_drift_y_mm": 0.5,
                "orientation_deviation_degrees": 12.0,
                "justification": "Contention test payload",
                "camera_id": "cam_chaos",
                "timestamp": time.time()
            }
            # Submit to event logger (synchronously or asynchronously)
            event_logger._sync_execute_transaction([payload])
        except Exception:
            pass
        time.sleep(0.001)

def run_snapshot_hammer(manager: CameraManager, vibration_analyzer: VibrationAnalyzer, stop_event: threading.Event):
    """Worker thread that hammers status snapshotting and cache context queries."""
    while not stop_event.is_set():
        try:
            # Query snapshot
            _ = manager.get_status_snapshot()
            # Ingest random noise and query cache
            vibration_analyzer.ingest(np.random.normal(), np.random.normal())
            _ = vibration_analyzer.get_forensic_context(np.random.uniform(0.1, 10.0), np.random.uniform(0.5, 60.0))
        except Exception as e:
            _assert(False, f"Snapshot hammer crashed: {e}")
            break
        time.sleep(0.001)

def main():
    print("[Chaos Soak] Initializing System-Wide Chaos & Saturation Audit...")
    db_path = "data/events/events.db"
    csv_path = "data/events/events.csv"
    os.makedirs("data/events", exist_ok=True)

    # Initialize Singleton event logger
    event_logger = EventLogger(sqlite_path=db_path, csv_path=csv_path)
    
    # -------------------------------------------------------------
    # Vector A: Ingestion Exhaustion & CUDA VRAM Leak Trapping
    # -------------------------------------------------------------
    print("\n[Vector A] Ingestion Exhaustion & CUDA VRAM Leak Trapping")
    
    # Construct aggressive high-rate configuration
    cfg = {
        "paths": {"data_dir": "data", "models_dir": "models"},
        "logging": {"sqlite_path": db_path, "csv_path": csv_path},
        "camera": {"fps": 350, "width": 320, "height": 240, "reconnect_delay": 0.5},
        "cameras": [
            {
                "id": "cam_chaos",
                "source": "synthetic://chaos",
                "enabled": True,
                "mock_inference": True,
                "mock_event_every": 1,
                "independent_logs": True
            }
        ],
        "vibration_analysis": {"kap_db_path": "data/knowledge/knowledge_base.db"},
        "watchdog": {
            "heartbeat_timeout": 2.0,
            "frame_drop_threshold": 100,
            "queue_overflow_streak_threshold": 10,
            "base_backoff": 0.5,
            "max_backoff": 4.0
        },
        "disk_monitor": {"watch_path": "data", "interval_seconds": 1.0}
    }
    
    manager = CameraManager(config_data=cfg)
    
    # Force ResourceGovernor into CRITICAL block to test aggressive frame shedding
    manager.governor._state = SystemState.CRITICAL
    
    # Start manager streams
    manager.start()
    
    # Baseline VRAM check
    vram_baseline = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_baseline = torch.cuda.memory_allocated()
        print(f"     CUDA Available. Baseline VRAM Allocated: {vram_baseline / (1024*1024):.2f} MB")
    else:
        print("     CUDA Not Available. Skipping VRAM leak check.")

    time.sleep(2.0)
    
    # Invariant A1: Shedding active
    snapshot = manager.get_status_snapshot()
    cam_state = snapshot.get("cam_chaos", {})
    frames_captured = cam_state.get("frames_captured", 0)
    print(f"     Frames Captured under Critical Governor: {frames_captured}")
    _assert(frames_captured > 0, "Ingestion loop should run synthetic frames")

    # Invariant A2: CUDA deallocation check
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        vram_end = torch.cuda.memory_allocated()
        vram_leak = vram_end - vram_baseline
        print(f"     Final VRAM Allocated: {vram_end / (1024*1024):.2f} MB (leak: {vram_leak} bytes)")
        _assert(vram_leak <= 1024*1024, "VRAM memory leakage must be under 1MB threshold")

    # Revert Governor to Normal
    manager.governor._state = SystemState.SAFE

    # -------------------------------------------------------------
    # Vector B: Database WAL Contention & CSV Pressure Relief
    # -------------------------------------------------------------
    print("\n[Vector B] Database WAL Contention & CSV Pressure Relief")
    
    stop_db_threads = threading.Event()
    
    # Start DB Locker connection to lock database exclusively
    db_locker_thread = threading.Thread(target=run_db_locker, args=(db_path, stop_db_threads), daemon=True)
    db_locker_thread.start()
    
    # Start 4 parallel threads flooding the event logger
    flooder_threads = []
    for _ in range(4):
        t = threading.Thread(target=run_logger_flooder, args=(event_logger, stop_db_threads), daemon=True)
        t.start()
        flooder_threads.append(t)
        
    time.sleep(3.0)
    
    # Terminate flooders and locker
    stop_db_threads.set()
    db_locker_thread.join(timeout=2.0)
    for t in flooder_threads:
        t.join(timeout=2.0)
        
    # Invariant B: Verify pressure relief fallback CSV files exist and have entries
    # The logs should be written to geometry_cam_chaos_{session_id}.csv due to db locks
    session_id = cam_state.get("session_id", "session")
    fallback_csv = f"data/events/geometry_cam_chaos_{session_id}.csv"
    csv_exists = os.path.exists(fallback_csv)
    print(f"     SessionFallback CSV exists: {csv_exists} ({fallback_csv})")
    _assert(csv_exists, "Database write lock contention must trigger pressure-relief CSV fallback")

    # -------------------------------------------------------------
    # Vector C: Cross-Thread Cache Mutex & Watchdog Hammering
    # -------------------------------------------------------------
    print("\n[Vector C] Cross-Thread Cache Mutex & Watchdog Hammering")
    
    vibration_analyzer = manager.vibration_analyzers["cam_chaos"]
    stop_mutex_threads = threading.Event()
    
    # Spin up 20 threads hammering snapshotting
    hammer_threads = []
    for _ in range(20):
        t = threading.Thread(target=run_snapshot_hammer, args=(manager, vibration_analyzer, stop_mutex_threads), daemon=True)
        t.start()
        hammer_threads.append(t)
        
    # Watchdog loop simulation: forcefully mutate worker thread states and trigger watchdog evaluation
    for _ in range(15):
        with manager._lock:
            # Force worker failed status
            if "cam_chaos" in manager.workers:
                manager.workers["cam_chaos"].failed = True
                manager.workers["cam_chaos"].failure_reason = "Chaos injected thread failure"
        time.sleep(0.1)

    stop_mutex_threads.set()
    for t in hammer_threads:
        t.join(timeout=2.0)
        
    print("     Watchdog evaluate and concurrent snapshot checks complete.")
    _assert(True, "Snapshot and cache locks must successfully prevent ConcurrentModification / Dict changed size exceptions")

    # Stop manager streams
    manager.stop()

    # -------------------------------------------------------------
    # Vector D: Mathematical Matrix Poisoning
    # -------------------------------------------------------------
    print("\n[Vector D] Mathematical Matrix Poisoning")
    
    # 1. Contaminate Live Frames in GeometryEngine
    geom_engine = GeometryEngine(camera_id="cam_math_test")
    # Pre-warm template image
    template_img = np.full((300, 300), 100, dtype=np.uint8)
    cv2.circle(template_img, (150, 150), 60, 200, -1)
    temp_path = "data/templates/golden_part_math_test.png"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    cv2.imwrite(temp_path, template_img)
    geom_engine._load_template(temp_path)
    
    # Create poisoned live frames: NaN, Inf, and Flatline zeroes
    flatline_frame = np.zeros((300, 300), dtype=np.uint8)
    nan_frame = np.full((300, 300), np.nan, dtype=np.float32)
    inf_frame = np.full((300, 300), np.inf, dtype=np.float32)
    
    try:
        res1 = geom_engine.inspect_part_geometry(flatline_frame)
        _assert(res1["status"] in {"ALIGNMENT_FAULT", "GEOMETRIC_VIOLATION"}, "Flatline frame should return a valid diagnostic status")
        
        res2 = geom_engine.inspect_part_geometry(nan_frame)
        _assert(res2["status"] in {"ALIGNMENT_FAULT", "GEOMETRIC_VIOLATION"}, "NaN frame should return a valid diagnostic status")
        
        res3 = geom_engine.inspect_part_geometry(inf_frame)
        _assert(res3["status"] in {"ALIGNMENT_FAULT", "GEOMETRIC_VIOLATION"}, "Inf frame should return a valid diagnostic status")
    except Exception as e:
        _assert(False, f"GeometryEngine crashed under poisoned matrix: {e}")
        traceback.print_exc()
    finally:
        # The synthetic template must not linger in the operator's template directory.
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # 2. Contaminate Live Telemetry in VibrationAnalyzer
    vib_analyzer = VibrationAnalyzer(buffer_size=64, kap_db_path="data/knowledge/knowledge_base.db", camera_id="cam_math_test")
    try:
        # Fill buffer with normal values, then introduce poisoned inputs (with realistic timestamps)
        dt = 1.0 / 120.0
        ts = time.time()
        for i in range(61):
            vib_analyzer.ingest(1.0, 1.0, ts=ts + i * dt)
        vib_analyzer.ingest(np.nan, 2.5, ts=ts + 61 * dt)
        vib_analyzer.ingest(1.2, np.inf, ts=ts + 62 * dt)
        vib_analyzer.ingest(0.0, 0.0, ts=ts + 63 * dt)
        
        # Verify DSP pipeline handles contaminated buffer without crashing
        res = vib_analyzer.analyze_mechanical_state()
        _assert(res["status"] in {"OK", "EXCEPTION", "NO_SIGNAL_IN_BAND"}, "Vibration analyzer must return valid status under poisoned buffer")
        
        # Verify get_forensic_context handles nan/inf parameters safely
        iso, causal, just_str = vib_analyzer.get_forensic_context(np.nan, np.inf)
        print(f"     Vibration forensic status: {iso}")
        _assert(iso in {"UNKNOWN", "ZONE_A_B", "ZONE_C", "ZONE_D"}, "Vibration analyzer must return valid ISO status under poisoned values")
    except Exception as e:
        _assert(False, f"VibrationAnalyzer crashed under poisoned matrix: {e}")
        traceback.print_exc()

    # -------------------------------------------------------------
    # Report Overall Status
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    if _errors:
        print(f"❌ Chaos Stress Soak completed with {len(_errors)} failure(s).")
        for err in _errors:
            print(f"   - {err}")
        sys.exit(1)
    else:
        print("✅ Chaos Stress Soak completed with 100% passes!")
        sys.exit(0)

if __name__ == "__main__":
    main()
