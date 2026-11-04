"""Cycle sources: where process cycles come from.

Real machines expose live signal values (Modbus registers, OPC UA nodes) plus a cycle
(shot) counter. :class:`CycleAssembler` turns that polled stream into discrete
:class:`ProcessCycle` objects: every counter increment closes the current cycle and
the samples collected since the previous increment become its curves.

Sources:

* :class:`SimulatedCycleSource` - the injection-molding simulator in (scaled) real time.
* :class:`ModbusCycleSource`    - Modbus TCP polling (pymodbus).
* :class:`OpcUaCycleSource`     - OPC UA polling (asyncua).
"""
from __future__ import annotations

import asyncio
import collections
import logging
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from src.process.models import ProcessCycle
from src.process.simulator import InjectionMoldingSimulator, SimulatedOutcome

logger = logging.getLogger("QualyxusCycleSources")

CycleCallback = Callable[[ProcessCycle], None]


@dataclass
class SignalSpec:
    """One machine signal. ``kind`` = curve (sampled into the cycle curve) or scalar
    (summarised per cycle; numeric scalars use the mean of the cycle's samples)."""

    name: str
    kind: str = "curve"
    scale: float = 1.0
    offset: float = 0.0
    # Modbus addressing
    address: int | None = None
    register_type: str = "holding"   # holding | input
    data_type: str = "int16"         # int16 | uint16 | int32 | uint32 | float32
    word_order: str = "big"          # big | little (for 32-bit types)
    # OPC UA addressing
    node_id: str | None = None

    @classmethod
    def from_config(cls, name: str, payload: dict[str, Any]) -> "SignalSpec":
        return cls(
            name=name,
            kind=str(payload.get("kind", "curve")),
            scale=float(payload.get("scale", 1.0)),
            offset=float(payload.get("offset", 0.0)),
            address=None if payload.get("address") is None else int(payload["address"]),
            register_type=str(payload.get("register_type", "holding")),
            data_type=str(payload.get("data_type", "int16")),
            word_order=str(payload.get("word_order", "big")),
            node_id=payload.get("node_id"),
        )


class CycleAssembler:
    """Segments a polled signal stream into cycles using a monotonic cycle counter."""

    def __init__(
        self,
        machine_id: str,
        signals: list[SignalSpec],
        setpoint_names: list[str] | None = None,
        *,
        nominal_sample_rate_hz: float = 20.0,
        max_cycle_s: float = 600.0,
        min_samples: int = 3,
        line_id: str | None = None,
        source: str = "plc",
    ):
        self.machine_id = machine_id
        self.line_id = line_id
        self.source = source
        self.curve_names = [s.name for s in signals if s.kind == "curve"]
        self.scalar_names = [s.name for s in signals if s.kind == "scalar"]
        self.setpoint_names = list(setpoint_names or [])
        self.nominal_sample_rate_hz = float(nominal_sample_rate_hz)
        self.max_cycle_s = float(max_cycle_s)
        self.min_samples = int(min_samples)
        self._counter: int | None = None
        self._samples: list[tuple[float, dict[str, float]]] = []
        self._setpoints: dict[str, float] = {}

    def feed(self, ts: float, values: dict[str, float], counter: int | None) -> ProcessCycle | None:
        for name in self.setpoint_names:
            if name in values and values[name] is not None:
                self._setpoints[name] = float(values[name])

        completed = None
        if counter is not None and self._counter is not None and int(counter) != self._counter:
            if int(counter) < self._counter:
                logger.warning("[CycleAssembler] %s cycle counter went backwards (%s -> %s); "
                               "assuming a controller reset.", self.machine_id, self._counter, counter)
            completed = self._close(self._counter)
        if counter is not None:
            self._counter = int(counter)

        self._samples.append((float(ts), dict(values)))
        # A machine that idles without counting must not grow one endless "cycle".
        cutoff = float(ts) - self.max_cycle_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.pop(0)
        return completed

    def _close(self, counter: int) -> ProcessCycle | None:
        samples, self._samples = self._samples, []
        if len(samples) < self.min_samples:
            return None
        times = np.array([t for t, _ in samples])
        duration = float(times[-1] - times[0])
        rate = (len(samples) - 1) / duration if duration > 0 else self.nominal_sample_rate_hz
        curves = {}
        for name in self.curve_names:
            column = [v.get(name) for _, v in samples]
            curves[name] = np.array([np.nan if c is None else float(c) for c in column], dtype=np.float32)
        scalars: dict[str, Any] = {}
        for name in self.scalar_names:
            column = [v.get(name) for _, v in samples if v.get(name) is not None]
            if column:
                scalars[name] = round(float(np.mean(column)), 6)
        start_ts = float(times[0])
        return ProcessCycle(
            cycle_id=f"{self.machine_id}-{int(start_ts)}-{counter}",
            machine_id=self.machine_id,
            line_id=self.line_id,
            sequence=int(counter),
            start_ts=start_ts,
            end_ts=float(times[-1]),
            setpoints=dict(self._setpoints),
            scalars=scalars,
            curves=curves,
            sample_rate_hz=float(rate),
            source=self.source,
        )


class CycleSource:
    """Base class: runs a background thread and hands completed cycles to callbacks."""

    name = "base"

    def __init__(self):
        self._callbacks: list[CycleCallback] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def subscribe(self, callback: CycleCallback) -> None:
        self._callbacks.append(callback)

    def _emit(self, cycle: ProcessCycle) -> None:
        for callback in list(self._callbacks):
            try:
                callback(cycle)
            except Exception as e:
                logger.error("[%s] Cycle callback failed for %s: %s", self.name, cycle.cycle_id, e, exc_info=True)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_safe, name=f"cycle-source-{self.name}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _run_safe(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                self._run()
                return
            except Exception as e:
                if time.monotonic() - started > 60.0:  # it was healthy: this is a new outage
                    backoff = 1.0
                logger.error("[%s] Source failed: %s. Reconnecting in %.0fs.", self.name, e, backoff)
                if self._stop.wait(backoff):
                    return
                backoff = min(backoff * 2.0, 30.0)

    def _run(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class SimulatedCycleSource(CycleSource):
    """Runs the simulator against the wall clock (optionally sped up by ``time_scale``).

    Ground-truth outcomes are retained so a synthetic camera can render each part as the
    process actually produced it (see :mod:`src.process.synthetic_parts`).
    """

    name = "simulated"

    def __init__(self, simulator: InjectionMoldingSimulator, *, time_scale: float = 1.0, history: int = 512):
        super().__init__()
        self.simulator = simulator
        self.time_scale = max(0.01, float(time_scale))
        self.outcomes: "collections.OrderedDict[str, tuple[ProcessCycle, SimulatedOutcome]]" = collections.OrderedDict()
        self._history = int(history)
        self._lock = threading.Lock()

    def _run(self) -> None:
        next_start = time.time()
        while not self._stop.is_set():
            cycle, outcome = self.simulator.next_cycle(start_ts=0.0)
            duration = cycle.duration_s / self.time_scale
            cycle.start_ts = next_start
            cycle.end_ts = next_start + duration
            if self._stop.wait(max(0.0, cycle.end_ts - time.time())):
                return
            with self._lock:
                self.outcomes[cycle.cycle_id] = (cycle, outcome)
                while len(self.outcomes) > self._history:
                    self.outcomes.popitem(last=False)
            self._emit(cycle)
            next_start = cycle.end_ts + 1.5 / self.time_scale  # mold open / eject / close

    def outcomes_ended_between(self, t0: float, t1: float) -> list[tuple[ProcessCycle, SimulatedOutcome]]:
        with self._lock:
            return [(c, o) for c, o in self.outcomes.values() if t0 <= c.end_ts < t1]

    def apply_setpoints(self, changes: dict[str, float]) -> dict[str, float]:
        return self.simulator.apply_setpoints(changes)


def _decode_registers(words: list[int], spec: SignalSpec) -> float:
    dtype = spec.data_type
    if dtype in ("int16", "uint16"):
        raw = int(words[0]) & 0xFFFF
        value = raw - 0x10000 if dtype == "int16" and raw >= 0x8000 else raw
    else:
        hi, lo = (words[0], words[1]) if spec.word_order == "big" else (words[1], words[0])
        packed = struct.pack(">HH", int(hi) & 0xFFFF, int(lo) & 0xFFFF)
        fmt = {"int32": ">i", "uint32": ">I", "float32": ">f"}[dtype]
        value = struct.unpack(fmt, packed)[0]
    return float(value) * spec.scale + spec.offset


class ModbusCycleSource(CycleSource):
    name = "modbus"

    def __init__(
        self,
        host: str,
        *,
        port: int = 502,
        unit_id: int = 1,
        signals: list[SignalSpec],
        counter: SignalSpec,
        setpoints: list[SignalSpec] | None = None,
        poll_hz: float = 20.0,
        machine_id: str = "machine_1",
        line_id: str | None = None,
        client_factory: Callable[[], Any] | None = None,
    ):
        super().__init__()
        self.host, self.port, self.unit_id = host, int(port), int(unit_id)
        self.signals = signals
        self.counter = counter
        self.setpoints = list(setpoints or [])
        self.poll_interval = 1.0 / max(0.5, float(poll_hz))
        self.assembler = CycleAssembler(
            machine_id, signals, [s.name for s in self.setpoints],
            nominal_sample_rate_hz=poll_hz, line_id=line_id, source="modbus",
        )
        self._client_factory = client_factory

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from pymodbus.client import ModbusTcpClient

        return ModbusTcpClient(self.host, port=self.port, timeout=1.0, retries=1)

    def _read(self, client, spec: SignalSpec) -> float:
        count = 1 if spec.data_type in ("int16", "uint16") else 2
        reader = client.read_input_registers if spec.register_type == "input" else client.read_holding_registers
        try:
            response = reader(spec.address, count=count, device_id=self.unit_id)
        except TypeError:  # pymodbus < 3.10
            response = reader(spec.address, count=count, slave=self.unit_id)
        if response is None or (hasattr(response, "isError") and response.isError()):
            raise IOError(f"Modbus read failed for {spec.name} @ {spec.address}: {response}")
        return _decode_registers(list(response.registers), spec)

    def _run(self) -> None:
        client = self._client()
        if not client.connect():
            raise ConnectionError(f"Cannot connect to Modbus server {self.host}:{self.port}")
        logger.info("[Modbus] Connected to %s:%s (unit %s)", self.host, self.port, self.unit_id)
        try:
            while not self._stop.is_set():
                tick = time.monotonic()
                values = {s.name: self._read(client, s) for s in self.signals + self.setpoints}
                counter = int(self._read(client, self.counter))
                cycle = self.assembler.feed(time.time(), values, counter)
                if cycle is not None:
                    self._emit(cycle)
                self._stop.wait(max(0.0, self.poll_interval - (time.monotonic() - tick)))
        finally:
            client.close()


class OpcUaCycleSource(CycleSource):
    name = "opcua"

    def __init__(
        self,
        endpoint: str,
        *,
        signals: list[SignalSpec],
        counter: SignalSpec,
        setpoints: list[SignalSpec] | None = None,
        poll_hz: float = 20.0,
        machine_id: str = "machine_1",
        line_id: str | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        super().__init__()
        self.endpoint = endpoint
        self.signals = signals
        self.counter = counter
        self.setpoints = list(setpoints or [])
        self.poll_interval = 1.0 / max(0.5, float(poll_hz))
        self.username, self.password = username, password
        self.assembler = CycleAssembler(
            machine_id, signals, [s.name for s in self.setpoints],
            nominal_sample_rate_hz=poll_hz, line_id=line_id, source="opcua",
        )

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        from asyncua import Client

        client = Client(url=self.endpoint, timeout=4)
        if self.username:
            client.set_user(self.username)
            client.set_password(self.password or "")
        async with client:
            logger.info("[OPC UA] Connected to %s", self.endpoint)
            specs = self.signals + self.setpoints + [self.counter]
            nodes = [client.get_node(s.node_id) for s in specs]
            while not self._stop.is_set():
                tick = time.monotonic()
                raw = await client.read_values(nodes)
                values = {s.name: float(v) * s.scale + s.offset for s, v in zip(specs, raw)}
                counter = int(values.pop(self.counter.name))
                cycle = self.assembler.feed(time.time(), values, counter)
                if cycle is not None:
                    self._emit(cycle)
                await asyncio.sleep(max(0.0, self.poll_interval - (time.monotonic() - tick)))


def build_cycle_source(process_cfg: dict[str, Any]) -> CycleSource:
    """Factory from the ``process:`` config section."""
    kind = str(process_cfg.get("source", "simulated")).lower()
    machine_id = str(process_cfg.get("machine_id", "imm_1"))
    line_id = process_cfg.get("line_id")
    if kind == "simulated":
        sim_cfg = process_cfg.get("simulator", {}) or {}
        simulator = InjectionMoldingSimulator(
            machine_id=machine_id,
            seed=int(sim_cfg.get("seed", 7)),
            disturbance_scale=float(sim_cfg.get("disturbance_scale", 1.0)),
            line_id=line_id,
        )
        return SimulatedCycleSource(simulator, time_scale=float(sim_cfg.get("time_scale", 1.0)))

    conn = process_cfg.get(kind, {}) or {}
    signals = [SignalSpec.from_config(n, p) for n, p in (conn.get("signals") or {}).items()]
    setpoints = [SignalSpec.from_config(n, {**p, "kind": "setpoint"}) for n, p in (conn.get("setpoints") or {}).items()]
    if not signals or not conn.get("cycle_counter"):
        raise ValueError(f"process.{kind} needs 'signals' and 'cycle_counter' definitions")
    counter = SignalSpec.from_config("cycle_counter", {**conn["cycle_counter"], "kind": "counter"})
    poll_hz = float(conn.get("poll_hz", 20.0))
    if kind == "modbus":
        return ModbusCycleSource(
            str(conn.get("host", "127.0.0.1")), port=int(conn.get("port", 502)), unit_id=int(conn.get("unit_id", 1)),
            signals=signals, counter=counter, setpoints=setpoints, poll_hz=poll_hz,
            machine_id=machine_id, line_id=line_id,
        )
    if kind == "opcua":
        return OpcUaCycleSource(
            str(conn["endpoint"]), signals=signals, counter=counter, setpoints=setpoints, poll_hz=poll_hz,
            machine_id=machine_id, line_id=line_id, username=conn.get("username"), password=conn.get("password"),
        )
    raise ValueError(f"Unknown process.source '{kind}' (expected simulated | modbus | opcua)")
