from __future__ import annotations
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
    endpoint: str | None = None
    unit_id: int | None = None


class TelemetryProvider:
    """The PLC connection, and the station context stored with each event."""

    def __init__(
        self,
        simulation_mode: bool = True,
        mode: str | None = None,
        protocol: str = "modbus",
        endpoint: str | None = None,
        port: int = 502,
        unit_id: int = 1,
        register_map: dict[str, Any] | None = None,
        require_ack: bool = False,
        min_verdict_hold_s: float = 0.1,
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
                require_ack=require_ack,
                min_verdict_hold_s=min_verdict_hold_s,
            )
        )
    def get_telemetry(self, camera_id: str) -> TelemetryContext:
        """Station context. Process values come from the machine cycle the part is linked to."""
        return TelemetryContext(**self.adapter.read_context(camera_id))
