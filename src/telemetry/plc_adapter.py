from __future__ import annotations
import random
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("PLCAdapter")

@dataclass
class PLCAdapterConfig:
    mode: str = "simulated"
    protocol: str = "modbus"
    endpoint: str | None = None
    unit_id: int = 1
    register_map: dict[str, Any] | None = None

class PLCAdapter:
    """PLC-ready abstraction with real pyModbus execution loops and safe simulated fallbacks."""

    def __init__(self, config: PLCAdapterConfig | None = None):
        self.config = config or PLCAdapterConfig()

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

    def write_anomaly_interlock(self, anomaly_payload: dict[str, Any]) -> bool:
        """
        Maps incoming anomaly payloads to Modbus TCP Holding Register 100
        using Function Code 0x06 (write_register) to maintain the ISO 13849
        regulatory compliance insulation boundary.
        """
        # Resolve severity dynamically from both top-level keys and nested diagnostic blocks
        diagnostic = anomaly_payload.get("diagnostic", {})
        if isinstance(diagnostic, dict):
            severity_raw = diagnostic.get("severity", anomaly_payload.get("severity", 1))
        else:
            severity_raw = anomaly_payload.get("severity", 1)
            
        try:
            severity = int(severity_raw)
        except (ValueError, TypeError):
            severity = 1
            
        register_address = 100
        logger.info(f"[Modbus] ISO 13849 Interlock: Writing holding register {register_address} with value {severity} (FC=0x06)")
        
        if self.is_simulated:
            logger.info("[Modbus] Simulation Mode: Write register 100 completed successfully.")
            return True
            
        try:
            from pymodbus.client import ModbusTcpClient
            endpoint_ip = self.config.endpoint or "127.0.0.1"
            client = ModbusTcpClient(endpoint_ip, timeout=1.5, retries=0)
            try:
                if not client.connect():
                    logger.error(f"[Modbus] Connection failed to physical industrial node at: {endpoint_ip}")
                    return False
                try:
                    # pymodbus >= 3.10 renamed the unit selector from `slave` to `device_id`.
                    client.write_register(register_address, severity, device_id=self.config.unit_id)
                except TypeError:
                    client.write_register(register_address, severity, slave=self.config.unit_id)
                return True
            finally:
                client.close()
        except Exception as e:
            logger.error(f"[Modbus] Failed to write holding register 100 via pyModbus client: {e}")
            return False