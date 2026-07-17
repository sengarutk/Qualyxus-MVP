from __future__ import annotations

import logging
import threading
import time
import psutil

from src.config.config import Config
from src.logging.schema import SystemState, SystemVitals


LOGGER = logging.getLogger(__name__)

try:  # Optional GPU telemetry
    import pynvml  # type: ignore
except Exception:  # pragma: no cover
    pynvml = None

class ResourceGovernor:
    """Core guardian module for SAFE/WARNING/CRITICAL resource assessment."""

    def __init__(
        self,
        config: Config | None = None,
    ):
        self.config = config or Config.load()
        self.rm = self.config.resource_management
        self._latency_samples: list[float] = []
        self._lock = threading.Lock()
        self._last_state = SystemState.SAFE
        self._last_transition_ts = time.time()
        self._critical_since: float | None = None
        self._warning_since: float | None = None
        # Stability knobs (seconds). These may be overridden via Config.resource_management.
        self._critical_hold_seconds = float(getattr(self.rm, "critical_hold_seconds", 300.0) or 300.0)
        self._warning_hold_seconds = float(getattr(self.rm, "warning_hold_seconds", 120.0) or 120.0)
        self._critical_cooldown_seconds = float(getattr(self.rm, "critical_cooldown_seconds", 900.0) or 900.0)
        self._warning_cooldown_seconds = float(getattr(self.rm, "warning_cooldown_seconds", 300.0) or 300.0)
        # Emergency threshold (immediate latch). Default chosen to match malicious-audit test.
        self._emergency_temp_c = float(getattr(self.rm, "emergency_temp_c", 90.0) or 90.0)

        # External hook: caller may register a callback invoked on EMERGENCY latch.
        self._emergency_callback = None

        self._nvml_ready = False
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_ready = True
            except Exception as exc:
                self._nvml_ready = False
                LOGGER.warning("pynvml unavailable, GPU vitals disabled: %s", exc)
        else:
            LOGGER.warning("pynvml not installed, GPU vitals default to 0")

    def record_inference_latency(self, latency_ms: float) -> None:
        try:
            with self._lock:
                self._latency_samples.append(float(latency_ms))
                if len(self._latency_samples) > 30:
                    self._latency_samples = self._latency_samples[-30:]
        except Exception:
            pass

    def allow_gpu_preprocess(self, vram_soft_limit: float = 90.0) -> bool:
        """Return False when VRAM is too high to safely run extra GPU kernels.

        This is a lightweight gate for high-frequency preprocessing kernels.
        If VRAM is already above the soft limit, callers should fall back to CPU
        preprocessing to avoid pushing the system into CRITICAL/EMERGENCY.
        """
        try:
            vitals = self.get_vitals()
            return float(vitals.vram_percent) <= float(vram_soft_limit)
        except Exception:
            return True

    def _cpu_temperature(self) -> float:
        try:
            temps = psutil.sensors_temperatures()
            for entries in temps.values():
                for entry in entries:
                    current = getattr(entry, "current", None)
                    if current is not None:
                        return float(current)
        except Exception as exc:
            LOGGER.warning("CPU temperature read failed: %s", exc)
            return 0.0
        return 0.0

    def _gpu_metrics(self) -> tuple[float, float]:
        if not self._nvml_ready:
            return 0.0, 0.0
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_pct = (float(mem.used) / float(mem.total) * 100.0) if mem.total else 0.0
            return temp, used_pct
        except Exception as exc:
            LOGGER.warning("GPU vitals read failed: %s", exc)
            return 0.0, 0.0

    def get_vitals(self) -> SystemVitals:
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
        except Exception as exc:
            LOGGER.warning("CPU load read failed: %s", exc)
            cpu_percent = 0.0
        try:
            ram_percent = float(psutil.virtual_memory().percent)
        except Exception as exc:
            LOGGER.warning("RAM usage read failed: %s", exc)
            ram_percent = 0.0
        cpu_temp = self._cpu_temperature()
        gpu_temp, vram_percent = self._gpu_metrics()
        return SystemVitals(
            cpu_load_percent=cpu_percent,
            ram_percent=ram_percent,
            cpu_temperature_c=cpu_temp,
            gpu_temperature_c=gpu_temp,
            vram_percent=vram_percent,
            timestamp=time.time(),
        )

    def get_policy_state(self) -> SystemState:
        vitals = self.get_vitals()
        now = time.time()
        emergency_hit = (
            vitals.cpu_temperature_c >= self._emergency_temp_c
            or vitals.gpu_temperature_c >= self._emergency_temp_c
        )
        critical_hit = (
            vitals.cpu_temperature_c >= self.rm.temp_thresholds.critical
            or vitals.gpu_temperature_c >= self.rm.temp_thresholds.critical
            or vitals.vram_percent >= self.rm.vram_thresholds.critical
            or vitals.ram_percent >= 95.0
        )
        warning_hit = (
            vitals.cpu_temperature_c >= self.rm.temp_thresholds.warning
            or vitals.gpu_temperature_c >= self.rm.temp_thresholds.warning
            or vitals.vram_percent >= self.rm.vram_thresholds.warning
            or vitals.ram_percent >= 85.0
        )

        if emergency_hit:
            candidate = SystemState.EMERGENCY
        elif critical_hit:
            candidate = SystemState.CRITICAL
        elif warning_hit:
            candidate = SystemState.WARNING
        else:
            candidate = SystemState.SAFE

        with self._lock:
            if candidate == SystemState.EMERGENCY:
                # Immediate latch; no hold-time. We do not auto-deescalate.
                if self._last_state != SystemState.EMERGENCY:
                    self._last_state = SystemState.EMERGENCY
                    self._last_transition_ts = now
                    try:
                        cb = self._emergency_callback
                        if cb is not None:
                            cb(vitals)
                    except Exception:
                        pass
                return self._last_state

            if critical_hit:
                self._critical_since = self._critical_since or now
            else:
                self._critical_since = None

            if warning_hit:
                self._warning_since = self._warning_since or now
            else:
                self._warning_since = None

            current = self._last_state

            if candidate == SystemState.CRITICAL:
                if current != SystemState.CRITICAL:
                    if self._critical_since is not None and (now - self._critical_since) >= self._critical_hold_seconds:
                        self._last_state = SystemState.CRITICAL
                        self._last_transition_ts = now
                return self._last_state

            if current == SystemState.CRITICAL:
                if candidate == SystemState.WARNING:
                    if (now - self._last_transition_ts) >= self._critical_cooldown_seconds:
                        self._last_state = SystemState.WARNING
                        self._last_transition_ts = now
                elif candidate == SystemState.SAFE:
                    if (now - self._last_transition_ts) >= self._critical_cooldown_seconds:
                        self._last_state = SystemState.WARNING
                        self._last_transition_ts = now
                return self._last_state

            if current == SystemState.EMERGENCY:
                # Emergency is latched until operator intervention / restart.
                return self._last_state

            if candidate == SystemState.WARNING:
                if current != SystemState.WARNING:
                    if self._warning_since is not None and (now - self._warning_since) >= self._warning_hold_seconds:
                        self._last_state = SystemState.WARNING
                        self._last_transition_ts = now
                return self._last_state

            if current == SystemState.WARNING and candidate == SystemState.SAFE:
                if (now - self._last_transition_ts) >= self._warning_cooldown_seconds:
                    self._last_state = SystemState.SAFE
                    self._last_transition_ts = now

            return self._last_state

    def set_emergency_callback(self, callback) -> None:
        """Register a callback invoked exactly once when transitioning into EMERGENCY."""
        with self._lock:
            self._emergency_callback = callback
