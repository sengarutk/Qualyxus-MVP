from __future__ import annotations
import random
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("PLCAdapter")

# Advisory holding registers. The plant's PLC logic decides what to do with every value.
#   verdict    0 = pass, 1-3 = reject by severity, 4 = part could not be inspected
#   part       machine cycle counter of that part (16 bit); it changes with every verdict
#   heartbeat  counts while inspection is healthy; if it stops, treat parts as not inspected
#   alert      machine-level alert (geometry, vibration) that belongs to no single part
PLC_REGISTERS = {"verdict": 100, "part": 101, "heartbeat": 102, "alert": 103}
PLC_PASS = 0
PLC_FLAG_NOT_INSPECTED = 4

@dataclass
class PLCAdapterConfig:
    mode: str = "simulated"
    protocol: str = "modbus"
    endpoint: str | None = None
    port: int = 502
    unit_id: int = 1
    register_map: dict[str, Any] | None = None

class PLCAdapter:
    """PLC-ready abstraction with real pyModbus execution loops and safe simulated fallbacks."""

    def __init__(self, config: PLCAdapterConfig | None = None):
        self.config = config or PLCAdapterConfig()
        self.registers = {**PLC_REGISTERS, **(self.config.register_map or {})}
        self.write_failures = 0
        self._heartbeat = 0
        self._client = None
        self._lock = threading.Lock()

    @property
    def is_simulated(self) -> bool:
        return str(self.config.mode).lower() == "simulated"

    def read_context(self, camera_id: str) -> dict[str, Any]:
        if self.is_simulated:
            return self._simulated_context(camera_id)
        return self._plc_placeholder_context(camera_id)

    def _simulated_context(self, camera_id: str) -> dict[str, Any]:
        base_temp = 75.0
        base_vib = 0.3
        base_speed = 1000.0
        temperature = max(60.0, min(90.0, base_temp + random.uniform(-15, 15)))
        vibration = max(0.1, min(0.5, base_vib + random.uniform(-0.2, 0.2)))
        speed = max(800.0, min(1200.0, base_speed + random.uniform(-200, 200)))
        suffix = camera_id.split("_")[-1]
        return {
            "machine_id": f"machine_{suffix}",
            "line_id": "line_A",
            "station_id": f"station_{suffix}",
            "temperature": round(temperature, 1),
            "vibration": round(vibration, 3),
            "speed": round(speed, 1),
            "lot_id": f"lot_{random.randint(1000, 9999)}",
            "shift_id": "day" if random.random() > 0.5 else "night",
            "telemetry_mode": "simulated",
            "plc_protocol": self.config.protocol,
        }

    def _plc_placeholder_context(self, camera_id: str) -> dict[str, Any]:
        suffix = camera_id.split("_")[-1]
        return {
            "machine_id": f"machine_{suffix}",
            "line_id": "line_A",
            "station_id": f"station_{suffix}",
            "temperature": None,
            "vibration": None,
            "speed": None,
            "lot_id": None,
            "shift_id": None,
            "telemetry_mode": "plc",
            "plc_protocol": self.config.protocol,
            "plc_ready": True,
            "opcua_compatible": True,
            "modbus_ready": True,
            "endpoint": self.config.endpoint,
            "unit_id": self.config.unit_id,
        }

    def _send(self, address: int, values: list[int]) -> bool:
        """Write over one persistent connection, reconnecting once if the PLC dropped it."""
        if self.is_simulated:
            return True
        with self._lock:
            for _attempt in range(2):
                try:
                    if self._client is None:
                        from pymodbus.client import ModbusTcpClient

                        client = ModbusTcpClient(self.config.endpoint or "127.0.0.1", port=self.config.port,
                                                 timeout=1.5, retries=0)
                        if not client.connect():
                            raise ConnectionError(f"cannot reach {self.config.endpoint}:{self.config.port}")
                        self._client = client
                    write = self._client.write_register if len(values) == 1 else self._client.write_registers
                    payload = values[0] if len(values) == 1 else values
                    try:
                        # pymodbus >= 3.10 renamed the unit selector from `slave` to `device_id`.
                        response = write(address, payload, device_id=self.config.unit_id)
                    except TypeError:
                        response = write(address, payload, slave=self.config.unit_id)
                    if hasattr(response, "isError") and response.isError():
                        raise IOError(str(response))
                    return True
                except Exception as e:
                    logger.warning(f"[Modbus] Write to register {address} failed: {e}")
                    self.close()
            self.write_failures += 1
            logger.error(f"[Modbus] Giving up on register {address}: {self.write_failures} failed writes so far.")
            return False

    def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                logger.debug("[Modbus] Client close failed", exc_info=True)

    def write_part_verdict(self, verdict: dict[str, Any]) -> bool:
        """One verdict for one physical part: flag, then the part number the PLC tracks.

        The part number changing tells the PLC a new verdict is ready. On adjacent registers
        both go in one Modbus frame, so the PLC can never read a flag with the wrong part.
        """
        flag, part = int(verdict["flag"]), int(verdict["part"]) & 0xFFFF
        logger.info(f"[Modbus] Part {part} {verdict.get('verdict')}: advisory flag {flag}")
        if self.registers["part"] == self.registers["verdict"] + 1:
            return self._send(self.registers["verdict"], [flag, part])
        return self._send(self.registers["verdict"], [flag]) and self._send(self.registers["part"], [part])

    def write_heartbeat(self) -> bool:
        """Counter the PLC watches: if it stops changing, treat parts as not inspected."""
        self._heartbeat = (self._heartbeat + 1) & 0xFFFF
        return self._send(self.registers["heartbeat"], [self._heartbeat])

    def write_advisory_flag(self, anomaly_payload: dict[str, Any]) -> bool:
        """Machine-level alert (geometry, vibration) that belongs to no single part."""
        diagnostic = anomaly_payload.get("diagnostic", {})
        if isinstance(diagnostic, dict):
            severity_raw = diagnostic.get("severity", anomaly_payload.get("severity", 1))
        else:
            severity_raw = anomaly_payload.get("severity", 1)
        try:
            severity = int(severity_raw)
        except (ValueError, TypeError):
            severity = 1
        kind = anomaly_payload.get("type", "ADVISORY")
        logger.info(f"[Modbus] Machine alert {kind} from {anomaly_payload.get('camera_id', 'unknown')}: {severity}")
        return self._send(self.registers["alert"], [severity])

    def clear(self) -> bool:
        """Leave no advisory flag set when the runtime stops."""
        return self._send(self.registers["verdict"], [PLC_PASS]) and self._send(self.registers["alert"], [PLC_PASS])
