from __future__ import annotations

import time
import logging
from dataclasses import dataclass

logger = logging.getLogger("QualyxusWatchdog")

@dataclass
class WatchdogDecision:
    restart: bool
    reason: str = ""
    backoff_seconds: float = 0.0

class WatchdogManager:
    def __init__(
        self,
        heartbeat_timeout: float = 10.0,
        frame_drop_threshold: int = 200,
        queue_overflow_streak_threshold: int = 25,
        base_backoff: float = 1.0,
        max_backoff: float = 30.0,
        logger_stall_timeout: float = 5.0,
        frame_drop_window_seconds: float = 60.0,
    ):
        self.heartbeat_timeout = heartbeat_timeout
        self.frame_drop_threshold = frame_drop_threshold
        self.queue_overflow_streak_threshold = queue_overflow_streak_threshold
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.logger_stall_timeout = logger_stall_timeout
        # Frame drops are judged per fixed-length window: a lifetime counter would eventually
        # restart a perfectly healthy long-running stream.
        self.frame_drop_window_seconds = max(1.0, float(frame_drop_window_seconds))

        self.failure_counts: dict[str, int] = {}
        self.next_restart_after: dict[str, float] = {}
        self._drop_baselines: dict[tuple[str, str], tuple[float, int]] = {}

    def _drops_in_window(self, state, cam: str, now: float) -> int:
        """Frames dropped by this worker session since the start of the current window."""
        dropped = int(getattr(state, "dropped_frames", 0) or 0)
        key = (cam, str(getattr(state, "session_id", "")))
        window_start, baseline = self._drop_baselines.get(key, (now, dropped))
        if dropped < baseline or (now - window_start) > self.frame_drop_window_seconds:
            window_start, baseline = now, dropped
        self._drop_baselines[key] = (window_start, baseline)
        return dropped - baseline

    def evaluate(self, state, now: float | None = None) -> WatchdogDecision:
        """
        Evaluates ingestion and storage layer telemetry profiles.
        Shielded against IEEE 754 float overflow casting limits and diagnostic log duplication.
        """
        now = now or time.perf_counter()
        cam = getattr(state, "camera_id", "unknown")

        # Hardening Fix: Move backoff cooling restrictions to the very top of the call.
        # Avoids wasting cycles running reflectance metrics or flooding log files during backoffs.
        next_allowed = self.next_restart_after.get(cam, 0.0)
        if now < next_allowed:
            return WatchdogDecision(restart=False)

        try:
            reason = None
            last_heartbeat = getattr(state, "last_heartbeat_ts", 0.0)
            
            if last_heartbeat > 0 and (now - last_heartbeat) > self.heartbeat_timeout:
                reason = "heartbeat_timeout"
            elif self._drops_in_window(state, cam, now) >= self.frame_drop_threshold:
                reason = "repeated_frame_drops"
            elif getattr(state, "queue_overflow_streak", 0) >= self.queue_overflow_streak_threshold:
                reason = "queue_overflow_persistent"
            
            logger_stall_ts = getattr(state, "logger_stall_start", 0.0)
            if logger_stall_ts > 0.0 and (now - logger_stall_ts) > self.logger_stall_timeout:
                logger.warning("[Watchdog] CRITICAL: Logger stall detected on %s for %.2f seconds.", cam, now - logger_stall_ts)
                reason = "logger_stall_timeout"
            elif getattr(state, "failed", False):
                reason = f"worker_failed:{getattr(state, 'failure_reason', 'unknown')}"

            if reason is None:
                return WatchdogDecision(restart=False)

            # A restarted worker gets a fresh drop window.
            for key in [k for k in self._drop_baselines if k[0] == cam]:
                self._drop_baselines.pop(key, None)

            # Process failure accumulation and handle backoff
            failures = self.failure_counts.get(cam, 0) + 1
            self.failure_counts[cam] = failures
            
            # Hardening Fix: Cap the exponent to a safe ceiling boundary (e.g., 30)
            # This completely blocks float conversion failures while easily satisfying max_backoff caps
            clamped_exponent = min(30, max(0, failures - 1))
            backoff = min(self.max_backoff, self.base_backoff * (2 ** clamped_exponent))
            self.next_restart_after[cam] = now + backoff

            return WatchdogDecision(restart=True, reason=reason, backoff_seconds=backoff)

        except Exception as e:
            logger.error("[Watchdog] Telemetry evaluation panic on %s. Shielding background thread: %s", cam, e)
            return WatchdogDecision(restart=False)

    def record_recovered(self, camera_id: str):
        """Resets structural error tracking maps upon a verified healthy execution cycle."""
        if self.failure_counts.get(camera_id, 0) > 0:
            logger.info("[Watchdog] Worker %s recovered from backoff penalty box. Structural state cleared.", camera_id)
        self.failure_counts[camera_id] = 0
        self.next_restart_after[camera_id] = 0.0
