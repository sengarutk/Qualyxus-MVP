from __future__ import annotations
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("PLCAdapter")

# Advisory holding registers. The plant's PLC logic decides what to do with every value.
#   verdict    0 = pass, 1-3 = reject by severity, 4 = part could not be inspected
#   part       machine cycle counter of that part (16 bit); 0 = the part matched no machine cycle
#   sequence   changes with every verdict, and is written last: act when it changes, never on
#              `part`, which repeats when a late reject updates a part or no cycle matched
#   ack        written by the PLC: copy `sequence` here once the verdict is taken. With
#              telemetry.require_ack no verdict is overwritten before that
#   cavity     which part of the shot (written only for multi-cavity stations)
#   heartbeat  counts while inspection is healthy; if it stops, treat parts as not inspected
#   alert      machine-level alert (geometry, vibration) that belongs to no single part
PLC_REGISTERS = {"verdict": 100, "part": 101, "heartbeat": 102, "alert": 103,
                 "sequence": 104, "ack": 105, "cavity": 106}
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
    require_ack: bool = False
    # Shortest time a verdict stays in the registers, so the PLC scans every one of them.
    min_verdict_hold_s: float = 0.1

class PLCAdapter:
    """PLC-ready abstraction with real pyModbus execution loops and safe simulated fallbacks."""

    def __init__(self, config: PLCAdapterConfig | None = None):
        self.config = config or PLCAdapterConfig()
        self.registers = {**PLC_REGISTERS, **(self.config.register_map or {})}
        self.write_failures = 0
        self._heartbeat = 0
        self._sequence: int | None = None      # last sequence the PLC holds; None until read from it
        self._awaiting_ack = True       # also after a restart: the last verdict may still be unread
        self._last_verdict_write = float("-inf")
        self._client = None
        self._lock = threading.Lock()

    @property
    def is_simulated(self) -> bool:
        return str(self.config.mode).lower() == "simulated"

    def read_context(self, camera_id: str) -> dict[str, Any]:
        """What is known about the station. Nothing measured means None: a value is never made up."""
        unknown = dict.fromkeys(("machine_id", "line_id", "station_id", "temperature", "vibration", "speed",
                                 "lot_id", "shift_id"))
        return {**unknown, "telemetry_mode": self.config.mode, "plc_protocol": self.config.protocol,
                **({} if self.is_simulated else {"endpoint": self.config.endpoint, "unit_id": self.config.unit_id})}

    def _connect(self):
        if self._client is None:
            from pymodbus.client import ModbusTcpClient

            client = ModbusTcpClient(self.config.endpoint or "127.0.0.1", port=self.config.port,
                                     timeout=1.5, retries=0)
            if not client.connect():
                raise ConnectionError(f"cannot reach {self.config.endpoint}:{self.config.port}")
            self._client = client
        return self._client

    def _call(self, method: str, address: int, *args, **kwargs):
        function = getattr(self._connect(), method)
        try:
            # pymodbus >= 3.10 renamed the unit selector from `slave` to `device_id`.
            response = function(address, *args, device_id=self.config.unit_id, **kwargs)
        except TypeError:
            response = function(address, *args, slave=self.config.unit_id, **kwargs)
        if hasattr(response, "isError") and response.isError():
            raise IOError(str(response))
        return response

    def _read(self, name: str) -> int | None:
        """One of our registers, or None if the PLC cannot be read."""
        with self._lock:
            for _attempt in range(2):
                try:
                    return int(self._call("read_holding_registers", self.registers[name], count=1).registers[0])
                except Exception as e:
                    logger.warning(f"[Modbus] Read of register {name} failed: {e}")
                    self.close()
            self.write_failures += 1
            return None

    def _send(self, address: int, values: list[int], deadline: float | None = None) -> bool:
        """Write over one persistent connection, reconnecting once if the PLC dropped it.

        After ``deadline`` (monotonic) nothing is sent: a late verdict is for a part that has gone.
        """
        if self.is_simulated:
            return True
        with self._lock:
            for _attempt in range(2):
                if deadline is not None and time.monotonic() >= deadline:
                    logger.warning(f"[Modbus] Register {address} not written: too late")
                    break
                try:
                    if len(values) == 1:
                        self._call("write_register", address, values[0])
                    else:
                        self._call("write_registers", address, values)
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

    def _registers_free(self, deadline: float | None) -> bool:
        """Wait until the previous verdict was seen: acknowledged, or held for a PLC scan."""
        hold = self._last_verdict_write + self.config.min_verdict_hold_s - time.monotonic()
        if hold > 0:
            time.sleep(hold)
        if not (self.config.require_ack and self._awaiting_ack):
            return True
        give_up = time.monotonic() + 3.0 if deadline is None else deadline
        while True:
            if self._sequence is None:
                self._sequence = self._read("sequence")
            if self._sequence is not None and self._read("ack") == self._sequence:
                self._awaiting_ack = False
                return True
            if time.monotonic() >= give_up:
                logger.error(f"[Modbus] PLC has not acknowledged verdict {self._sequence}; the next one is held back.")
                return False
            time.sleep(0.02)

    def _next_sequence(self) -> int | None:
        if self._sequence is None:
            # Continue from what the PLC holds, or a restart could repeat the number it last saw.
            self._sequence = self._read("sequence")
            if self._sequence is None:
                return None
        return self._sequence % 0xFFFF + 1     # 1..65535: never 0, the value of a fresh PLC

    def write_part_verdict(self, verdict: dict[str, Any], deadline: float | None = None) -> bool:
        """One verdict for one physical part: flag and part number, then the sequence.

        The PLC acts when the sequence changes. It is written last, so by then the flag and
        part are already there; on adjacent registers those two travel in one Modbus frame.
        """
        flag, part = int(verdict["flag"]), int(verdict["part"]) & 0xFFFF
        logger.info(f"[Modbus] Part {part} {verdict.get('verdict')}: advisory flag {flag}")
        if self.is_simulated:
            return True
        if not self._registers_free(deadline):
            return False
        sequence = self._next_sequence()
        if sequence is None:
            return False
        if self.registers["part"] == self.registers["verdict"] + 1:
            written = self._send(self.registers["verdict"], [flag, part], deadline)
        else:
            written = (self._send(self.registers["verdict"], [flag], deadline)
                       and self._send(self.registers["part"], [part], deadline))
        if written and verdict.get("cavity") is not None:
            written = self._send(self.registers["cavity"], [int(verdict["cavity"])], deadline)
        if not (written and self._send(self.registers["sequence"], [sequence], deadline)):
            # A lost reply may hide a write that did arrive: ask the PLC what it holds next time.
            self._sequence, self._awaiting_ack = None, True
            return False
        self._sequence, self._awaiting_ack = sequence, True
        self._last_verdict_write = time.monotonic()
        return True

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
        return self._send(self.registers["alert"], [max(0, min(severity, 0xFFFF))])

    def clear(self) -> bool:
        """Stopping: no alert stays set, and the verdict register must not read as a pass."""
        if self.is_simulated:
            return True
        if not self._registers_free(None):
            logger.error("[Modbus] Stopping with the last verdict unacknowledged.")
        return (self._send(self.registers["verdict"], [PLC_FLAG_NOT_INSPECTED])
                and self._send(self.registers["alert"], [0]))
