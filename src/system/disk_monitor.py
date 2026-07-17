import os
import shutil
import threading
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("QualyxusDiskMonitor")

@dataclass
class DiskPolicy:
    raw_enabled: bool = True
    defect_enabled: bool = True
    free_percent: float = 100.0
    level: str = "normal"
    write_latency_ms: float = 0.0

class DiskMonitor:
    """
    Thread-Safe non-blocking Disk Monitor and FIFO Sweeper.
    Keeps track of disk usage/policies and performs aggressive sweeps when thresholds are breached.
    """
    def __init__(self, watch_path: str = "data", interval_seconds: float = 30.0,
                 warning_threshold: float = 15.0, critical_threshold: float = 10.0):
        self.watch_path = Path(watch_path)
        self.interval_seconds = interval_seconds
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

        # High and low watermarks for sweeping (derived from warning and critical thresholds)
        self.high_watermark = (100.0 - warning_threshold) / 100.0
        self.low_watermark = (100.0 - warning_threshold - 15.0) / 100.0
        if self.low_watermark <= 0:
            self.low_watermark = 0.50

        self._policy = DiskPolicy()
        self._stop_event = threading.Event()
        self._worker_thread = None
        self._lock = threading.Lock()
        self._target_extensions = {".jpg", ".png", ".json"}

    def start(self):
        """Engages the background monitoring and sweeping thread."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._monitor_loop,
                name="Qualyxus-DiskMonitor",
                daemon=True
            )
            self._worker_thread.start()
            logger.info("[DiskMonitor] Engaged FIFO asset truncation engine on target: %s", self.watch_path)

    def stop(self):
        """Safely disengages the background thread."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

    def get_policy(self) -> DiskPolicy:
        with self._lock:
            return DiskPolicy(
                raw_enabled=self._policy.raw_enabled,
                defect_enabled=self._policy.defect_enabled,
                free_percent=self._policy.free_percent,
                level=self._policy.level,
                write_latency_ms=self._policy.write_latency_ms,
            )

    def allow_raw_save(self) -> bool:
        return self.get_policy().raw_enabled

    def allow_defect_save(self) -> bool:
        return self.get_policy().defect_enabled

    def _monitor_loop(self):
        # Initial boot delay to prevent IO storms during system startup
        if self._stop_event.wait(1.0):
            return
            
        while not self._stop_event.is_set():
            try:
                self._refresh_policy()
                self._check_and_sweep()
            except Exception as e:
                logger.error("[DiskMonitor] Unhandled exception in background sweep logic: %s", e)
                
            # Sleep for the designated interval while remaining interruptible
            self._stop_event.wait(self.interval_seconds)

    def _refresh_policy(self):
        simulated = os.getenv("SIMULATED_DISK_FREE_PERCENT")
        if simulated is not None:
            free_percent = float(simulated)
        elif os.getenv("SIMULATE_DISK_FULL") == "1":
            free_percent = 5.0
        else:
            self.watch_path.mkdir(parents=True, exist_ok=True)
            try:
                usage = shutil.disk_usage(str(self.watch_path))
                free_percent = (usage.free / usage.total) * 100.0 if usage.total > 0 else 0.0
            except Exception:
                free_percent = 100.0

        write_latency_ms = self._measure_write_latency()

        level = "normal"
        raw_enabled = True
        defect_enabled = True

        if free_percent < self.critical_threshold:
            level = "critical"
            raw_enabled = False
            defect_enabled = False  # Test expects false in critical mode
            logger.critical("[DiskMonitor] STORAGE CRITICAL: %.2f%% free. Disabling raw and defect saves.", free_percent)
        elif free_percent < self.warning_threshold:
            level = "warning"
            raw_enabled = False
            defect_enabled = True
            logger.warning("[DiskMonitor] STORAGE WARNING: %.2f%% free. Disabling raw saves.", free_percent)

        with self._lock:
            self._policy.free_percent = free_percent
            self._policy.level = level
            self._policy.raw_enabled = raw_enabled
            self._policy.defect_enabled = defect_enabled
            self._policy.write_latency_ms = write_latency_ms

    def _measure_write_latency(self) -> float:
        try:
            test_file = self.watch_path / ".disk_test.tmp"
            test_data = b"test" * 1024
            
            if os.getenv("SIMULATE_SLOW_WRITES") == "1":
                time.sleep(0.5)
                start_time = time.time() - 0.5
            else:
                start_time = time.time()
            
            with open(test_file, "wb") as f:
                f.write(test_data)
                f.flush()
                os.fsync(f.fileno())
            end_time = time.time()
            
            test_file.unlink(missing_ok=True)
            return (end_time - start_time) * 1000.0
        except Exception:
            return 999.0

    def _check_and_sweep(self):
        if not os.path.exists(self.watch_path):
            return

        try:
            total, used, free = shutil.disk_usage(str(self.watch_path))
            current_usage_pct = used / total if total > 0 else 0
        except Exception as e:
            logger.error("[DiskMonitor] Failed to read disk usage for sweep: %s", e)
            return

        # Trigger sweep if watermarks breached
        if current_usage_pct >= self.high_watermark:
            logger.warning(
                "[DiskMonitor] STORAGE CRITICAL: Usage %.1f%% breached %.1f%% watermark. Initiating aggressive FIFO sweep.", 
                current_usage_pct * 100, self.high_watermark * 100
            )
            self._execute_fifo_sweep(total, used)

    def _execute_fifo_sweep(self, total_bytes: int, current_used_bytes: int):
        target_used_bytes = int(total_bytes * self.low_watermark)
        bytes_to_reclaim = current_used_bytes - target_used_bytes

        if bytes_to_reclaim <= 0:
            return

        logger.info("[DiskMonitor] Reclaiming %.2f MB to restore storage homeostasis...", bytes_to_reclaim / (1024 * 1024))
        
        file_registry = []
        candidate_directories = set()
        base_watch_path = os.path.abspath(self.watch_path)
        try:
            for root, _, files in os.walk(self.watch_path):
                if self._stop_event.is_set():
                    return
                
                abs_root = os.path.abspath(root)
                if abs_root != base_watch_path:
                    candidate_directories.add(abs_root)
                    
                for file_name in files:
                    ext = os.path.splitext(file_name)[1].lower()
                    if ext in self._target_extensions:
                        full_path = os.path.join(root, file_name)
                        try:
                            stat = os.stat(full_path)
                            file_registry.append((full_path, stat.st_mtime, stat.st_size))
                        except (FileNotFoundError, PermissionError):
                            continue
        except Exception as e:
            logger.error("[DiskMonitor] Sweep directory walk failed: %s", e)
            return

        file_registry.sort(key=lambda x: x[1])

        bytes_reclaimed = 0
        files_deleted = 0

        for full_path, _, file_size in file_registry:
            if self._stop_event.is_set() or bytes_reclaimed >= bytes_to_reclaim:
                break

            try:
                os.remove(full_path)
                bytes_reclaimed += file_size
                files_deleted += 1
            except FileNotFoundError:
                pass
            except PermissionError:
                pass
            except Exception as e:
                logger.debug("Failed to prune asset %s: %s", full_path, e)

        # Cleanup parent folders
        sorted_candidates = sorted(list(candidate_directories), key=len, reverse=True)
        for d in sorted_candidates:
            if self._stop_event.is_set():
                break
            try:
                if os.path.exists(d) and not os.listdir(d):
                    os.rmdir(d)
            except OSError:
                pass

        logger.info("[DiskMonitor] Sweep Complete: Purged %d stale assets, recovered %.2f MB.", 
                    files_deleted, bytes_reclaimed / (1024 * 1024))
