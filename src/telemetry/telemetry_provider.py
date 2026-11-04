from __future__ import annotations
import threading
import time
from dataclasses import dataclass
from typing import Any

from src.telemetry.plc_adapter import PLCAdapter, PLCAdapterConfig


@dataclass
class TelemetryContext:
    machine_id: str | None = None
    line_id: str | None = None
    station_id: str | None = None
    temperature: float | None = None
    vibration: float | None = None
    speed: float | None = None
    lot_id: str | None = None
    shift_id: str | None = None
    telemetry_mode: str = "simulated"
    plc_protocol: str | None = None
    plc_ready: bool | None = None
    opcua_compatible: bool | None = None
    modbus_ready: bool | None = None
    endpoint: str | None = None
    unit_id: int | None = None


class TelemetryProvider:
    """
    Lightweight telemetry provider for manufacturing context.
    Supports both real sensor integration and simulation mode.
    """

    def __init__(
        self,
        simulation_mode: bool = True,
        mode: str | None = None,
        protocol: str = "modbus",
        endpoint: str | None = None,
        port: int = 502,
        unit_id: int = 1,
        register_map: dict[str, Any] | None = None,
    ):
        resolved_mode = mode or ("simulated" if simulation_mode else "plc")
        self.simulation_mode = resolved_mode == "simulated"
        self.mode = resolved_mode
        self.adapter = PLCAdapter(
            PLCAdapterConfig(
                mode=resolved_mode,
                protocol=protocol,
                endpoint=endpoint,
                port=port,
                unit_id=unit_id,
                register_map=register_map,
            )
        )
        self._sim_targets: dict[str, dict[str, Any]] = {}
        self._sim_lock = threading.Lock()

    def register_command_target(self, camera_id: str, metric_name: str, target_value: float, duration_seconds: float = 4.0):
        if self.mode != "simulated":
            return
        with self._sim_lock:
            self._sim_targets[camera_id] = {
                "metric_name": metric_name,
                "target_value": float(target_value),
                "started_at": time.time(),
                "duration_seconds": float(max(1.0, duration_seconds)),
            }

    def get_telemetry(self, camera_id: str) -> TelemetryContext:
        """
        Fetch current telemetry context for the given camera/machine.
        Returns simulated data if simulation_mode=True.
        """
        if self.mode == "simulated":
            return self._simulate_telemetry(camera_id)
        payload = self.adapter.read_context(camera_id)
        return TelemetryContext(**payload)

    def _simulate_telemetry(self, camera_id: str) -> TelemetryContext:
        """Generate realistic simulated telemetry ranges."""
        payload = self.adapter.read_context(camera_id)
        with self._sim_lock:
            target = self._sim_targets.get(camera_id)
        if target:
            metric_name = str(target.get("metric_name", "speed"))
            current_value = float(payload.get(metric_name, 0.0) or 0.0)
            target_value = float(target.get("target_value", current_value) or current_value)
            elapsed = max(0.0, time.time() - float(target.get("started_at", time.time())))
            duration = float(target.get("duration_seconds", 4.0) or 4.0)
            alpha = min(1.0, elapsed / max(duration, 1e-6))
            payload[metric_name] = round(current_value + ((target_value - current_value) * alpha), 3)
            if alpha >= 1.0:
                with self._sim_lock:
                    self._sim_targets.pop(camera_id, None)
        return TelemetryContext(**payload)