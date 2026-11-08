from __future__ import annotations

import numpy as np
import pytest

from src.process.alignment import CycleAligner, StationLink, estimate_transport_delay
from src.process.features import CycleFeatureExtractor, GoldenCycleReference, curve_statistics
from src.process.models import FAIL, PASS, InspectionRecord, ProcessCycle
from src.process.simulator import SETPOINT_LIMITS, InjectionMoldingSimulator
from src.process.sources import CycleAssembler, ModbusCycleSource, SignalSpec, _decode_registers
from src.process.store import ProcessStore


# ----------------------------------------------------------------- features

def test_curve_statistics_known_signal():
    fs = 10.0
    values = np.array([0, 1, 2, 4, 2, 0], dtype=float)
    stats = curve_statistics(values, fs)
    assert stats["max"] == 4
    assert stats["time_to_peak_s"] == pytest.approx(0.3)
    assert stats["integral"] == pytest.approx(np.trapezoid(values, dx=0.1))
    assert stats["max_slope"] == pytest.approx(20.0)
    assert stats["final"] == 0


def test_golden_reference_deviation_is_small_for_good_and_large_for_shifted():
    sim = InjectionMoldingSimulator(seed=3, start_ts=0.0, disturbance_scale=0.0)
    cycles = [sim.next_cycle()[0] for _ in range(30)]
    golden = GoldenCycleReference.fit(cycles, min_cycles=20)
    good = golden.deviation("hold_pressure", cycles[-1].curves["hold_pressure"])
    shifted = golden.deviation("hold_pressure", cycles[-1].curves["hold_pressure"] * 0.8)
    assert good["golden_rmse"] < shifted["golden_rmse"]
    assert shifted["golden_bias"] < 0
    roundtrip = GoldenCycleReference.from_dict(golden.to_dict())
    assert roundtrip.deviation("hold_pressure", cycles[0].curves["hold_pressure"]) == pytest.approx(
        golden.deviation("hold_pressure", cycles[0].curves["hold_pressure"])
    )


def test_feature_extractor_includes_setpoints_scalars_and_golden():
    sim = InjectionMoldingSimulator(seed=1, start_ts=0.0)
    cycles = [sim.next_cycle()[0] for _ in range(25)]
    features = CycleFeatureExtractor(GoldenCycleReference.fit(cycles, min_cycles=20)).extract(cycles[0])
    assert "setpoint.hold_pressure_sp" in features
    assert "cushion_mm" in features and "material_lot" not in features
    assert "melt_temp.golden_rmse" in features
    assert all(np.isfinite(v) for v in features.values())


# ---------------------------------------------------------------- simulator

def test_simulator_is_deterministic_per_seed():
    a = InjectionMoldingSimulator(seed=5, start_ts=0.0)
    b = InjectionMoldingSimulator(seed=5, start_ts=0.0)
    for _ in range(50):
        (ca, oa), (cb, ob) = a.next_cycle(), b.next_cycle()
        assert oa.defect_type == ob.defect_type
        np.testing.assert_allclose(ca.curves["injection_pressure"], cb.curves["injection_pressure"])


def test_simulator_defects_are_caused_by_process_state():
    sim = InjectionMoldingSimulator(seed=2, start_ts=0.0, disturbance_scale=0.0)
    healthy = [sim.next_cycle()[1] for _ in range(200)]
    sim.heater_offset = -16.0  # failed heater band: melt far too cold
    cold = [sim.next_cycle()[1] for _ in range(200)]
    healthy_rate = np.mean([o.defect_type == "short_shot" for o in healthy])
    cold_rate = np.mean([o.defect_type == "short_shot" for o in cold])
    assert healthy_rate < 0.02
    assert cold_rate > 0.8
    # The right setpoint correction removes the defect mechanism.
    sim.apply_setpoints({"melt_temp_sp": 246.0})
    fixed = [sim.next_cycle()[1] for _ in range(200)]
    assert np.mean([o.defect_type == "short_shot" for o in fixed]) < 0.05


def test_apply_setpoints_clips_and_ignores_unknown():
    sim = InjectionMoldingSimulator(seed=1)
    applied = sim.apply_setpoints({"melt_temp_sp": 999.0, "not_a_setpoint": 1.0})
    assert applied == {"melt_temp_sp": SETPOINT_LIMITS["melt_temp_sp"][1]}


# ------------------------------------------------------- store + alignment

def _cycle(seq: int, end_ts: float, machine: str = "m1") -> ProcessCycle:
    return ProcessCycle(cycle_id=ProcessCycle.make_id(machine, seq), machine_id=machine, sequence=seq,
                        start_ts=end_ts - 10.0, end_ts=end_ts, setpoints={"a": 1.0})


@pytest.fixture
def store(tmp_path):
    return ProcessStore(tmp_path / "process.db")


def test_store_roundtrip_with_curves(store):
    sim = InjectionMoldingSimulator(seed=1, start_ts=0.0)
    cycle, _ = sim.next_cycle()
    store.save_cycle(cycle, {"x": 1.0})
    loaded = store.get_cycle(cycle.cycle_id, with_curves=True)
    assert loaded.setpoints == cycle.setpoints
    np.testing.assert_allclose(loaded.curves["melt_temp"], cycle.curves["melt_temp"])
    assert store.get_features(cycle.cycle_id) == {"x": 1.0}


def test_time_alignment_picks_nearest_unlinked_cycle(store):
    for seq in range(1, 11):
        store.save_cycle(_cycle(seq, 100.0 + seq * 20.0), {})
    aligner = CycleAligner(store, [StationLink("cam", "m1", "time", transport_delay_s=7.0, tolerance_s=5.0)])
    # Part from cycle 4 (end 180) seen 7.4 s later.
    res = aligner.align("cam", 187.4)
    assert res.cycle_id == ProcessCycle.make_id("m1", 4)
    assert res.confidence == pytest.approx(1 - 0.4 / 5.0)
    assert aligner.align("cam", 500.0).cycle_id is None
    assert aligner.align("other_cam", 187.4).method == "unlinked"


def test_time_alignment_prefers_unlinked_when_ambiguous(store):
    store.save_cycle(_cycle(1, 100.0), {})
    store.save_cycle(_cycle(2, 103.0), {})
    aligner = CycleAligner(store, [StationLink("cam", "m1", "time", transport_delay_s=0.0, tolerance_s=5.0)])
    first = aligner.align("cam", 100.5)
    store.save_inspection(InspectionRecord("i1", "cam", 100.5, PASS, cycle_id=first.cycle_id))
    second = aligner.align("cam", 100.6)
    assert first.cycle_id != second.cycle_id


def test_fifo_and_part_id_alignment(store):
    for seq in range(1, 6):
        store.save_cycle(_cycle(seq, 100.0 + seq), {})
    fifo = CycleAligner(store, [StationLink("cam", "m1", "fifo", fifo_offset=2)])
    assert fifo.align("cam", 105.5).cycle_id == ProcessCycle.make_id("m1", 3)
    pid = CycleAligner(store, [StationLink("cam", "m1", "part_id")])
    assert pid.align("cam", 0.0, part_id=ProcessCycle.make_id("m1", 2)).confidence == 1.0
    assert pid.align("cam", 0.0, part_id="unknown").cycle_id is None


def test_transport_delay_estimation_recovers_delay_with_stoppages():
    rng = np.random.default_rng(0)
    ends, t = [], 0.0
    for i in range(120):
        t += 18.0 + rng.normal(0, 0.3) + (90.0 if i in (30, 75) else 0.0)  # two machine stops
        ends.append(t)
    seen = np.array(ends) + 42.5 + rng.normal(0, 0.2, len(ends))
    est = estimate_transport_delay(ends, seen)
    assert est["transport_delay_s"] == pytest.approx(42.5, abs=0.3)
    assert not est["ambiguous"]


def test_transport_delay_estimation_flags_periodic_ambiguity():
    ends = np.arange(1, 80) * 18.0
    est = estimate_transport_delay(ends, ends + 42.5)
    assert est["ambiguous"]


def test_labelled_dataset_uses_audit_over_vision(store):
    store.save_cycle(_cycle(1, 10.0), {"f": 1.0})
    store.save_cycle(_cycle(2, 20.0), {"f": 2.0})
    c1, c2 = ProcessCycle.make_id("m1", 1), ProcessCycle.make_id("m1", 2)
    store.save_inspection(InspectionRecord("a", "cam", 11, FAIL, defect_type="anomaly", cycle_id=c1))
    store.save_inspection(InspectionRecord("b", "audit", 12, PASS, cycle_id=c1, source="audit"))
    store.save_inspection(InspectionRecord("c", "cam", 21, FAIL, defect_type="flash", cycle_id=c2))
    rows = {r["cycle_id"]: r for r in store.labelled_dataset()}
    assert rows[c1]["label"] == 0           # audit overrides a false reject
    assert rows[c2]["label"] == 1 and rows[c2]["defect_type"] == "flash"


# ---------------------------------------------------------------- sources

def test_cycle_assembler_segments_on_counter_increment():
    asm = CycleAssembler("m1", [SignalSpec("p"), SignalSpec("cushion", kind="scalar")], ["sp"])
    out = []
    t = 0.0
    for counter in (5, 6, 7):
        for k in range(10):
            t += 0.1
            res = asm.feed(t, {"p": float(k), "cushion": 4.0, "sp": 600.0}, counter)
            if res is not None:
                out.append(res)
    assert [c.sequence for c in out] == [5, 6]
    assert len(out[1].curves["p"]) == 10
    assert out[1].scalars["cushion"] == 4.0 and out[1].setpoints == {"sp": 600.0}
    assert out[1].sample_rate_hz == pytest.approx(10.0)


def test_cycle_assembler_bounds_idle_samples():
    asm = CycleAssembler("m1", [SignalSpec("p")], max_cycle_s=5.0)
    for k in range(200):
        asm.feed(k * 0.1, {"p": 1.0}, 1)
    cycle = asm.feed(20.1, {"p": 1.0}, 2)
    assert cycle.duration_s <= 5.0 + 1e-6


def test_register_decoding():
    assert _decode_registers([0xFFFE], SignalSpec("x", data_type="int16", scale=0.5)) == -1.0
    import struct
    hi, lo = struct.unpack(">HH", struct.pack(">f", 231.5))
    assert _decode_registers([hi, lo], SignalSpec("x", data_type="float32")) == pytest.approx(231.5)
    assert _decode_registers([lo, hi], SignalSpec("x", data_type="float32", word_order="little")) == pytest.approx(231.5)


class _FakeModbusClient:
    def __init__(self):
        self.reads = 0

    def connect(self):
        return True

    def close(self):
        pass

    def read_holding_registers(self, address, count=1, device_id=1):
        self.reads += 1

        class _Resp:
            def __init__(self, regs):
                self.registers = regs

            def isError(self):
                return False
        if address == 100:  # cycle counter: increments every 5 polls
            return _Resp([self.reads // 15])
        return _Resp([address])


def test_modbus_source_emits_cycles_from_polled_registers():
    import time as _time

    client = _FakeModbusClient()
    source = ModbusCycleSource(
        "fake", signals=[SignalSpec("pressure", address=200, scale=0.1)],
        counter=SignalSpec("counter", address=100, data_type="uint16"),
        poll_hz=200.0, machine_id="m1", client_factory=lambda: client,
    )
    cycles = []
    source.subscribe(cycles.append)
    source.start()
    deadline = _time.time() + 5.0
    while len(cycles) < 2 and _time.time() < deadline:
        _time.sleep(0.02)
    source.stop()
    assert len(cycles) >= 2
    assert cycles[0].curves["pressure"][0] == pytest.approx(20.0)


def test_opcua_source_reads_live_server():
    """Round trip against a real (in-process) OPC UA server."""
    import asyncio
    import socket
    import threading
    import time as _time

    from asyncua import Server, ua

    from src.process.sources import OpcUaCycleSource

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"opc.tcp://127.0.0.1:{port}/qualyxus/"
    ready, stop = threading.Event(), threading.Event()

    async def serve():
        server = Server()
        await server.init()
        server.set_endpoint(endpoint)
        idx = await server.register_namespace("urn:qualyxus:test")
        machine = await server.nodes.objects.add_object(idx, "Machine")
        pressure = await machine.add_variable(ua.NodeId("Pressure", idx), "Pressure", 0.0)
        counter = await machine.add_variable(ua.NodeId("Counter", idx), "Counter", 0)
        async with server:
            ready.set()
            k = 0
            while not stop.is_set():
                k += 1
                await pressure.write_value(float(k))
                await counter.write_value(ua.Variant(k // 10, ua.VariantType.Int64))
                await asyncio.sleep(0.01)

    thread = threading.Thread(target=lambda: asyncio.run(serve()), daemon=True)
    thread.start()
    assert ready.wait(15)

    source = OpcUaCycleSource(
        endpoint, signals=[SignalSpec("pressure", node_id="ns=2;s=Pressure")],
        counter=SignalSpec("counter", node_id="ns=2;s=Counter"), poll_hz=100.0, machine_id="m1",
    )
    cycles = []
    source.subscribe(cycles.append)
    source.start()
    deadline = _time.time() + 15.0
    while len(cycles) < 2 and _time.time() < deadline:
        _time.sleep(0.05)
    source.stop()
    stop.set()
    thread.join(timeout=10)
    assert len(cycles) >= 2
    assert np.all(np.diff(cycles[-1].curves["pressure"]) >= 0)


def _store_with_cycle(tmp_path, end_ts=100.0):
    from src.process.models import ProcessCycle

    s = ProcessStore(tmp_path / "links.db")
    s.save_cycle(ProcessCycle(cycle_id="c1", machine_id="m", sequence=1, start_ts=end_ts - 20, end_ts=end_ts,
                              setpoints={}, scalars={}, curves={}, sample_rate_hz=20.0), {"x": 1.0})
    return s


def test_multi_cavity_shot_links_every_part_to_the_same_cycle(tmp_path):
    from src.process.alignment import CycleAligner, StationLink
    from src.process.models import InspectionRecord

    store = _store_with_cycle(tmp_path)
    aligner = CycleAligner(store, [StationLink("cam", "m", transport_delay_s=4.0, tolerance_s=2.0, parts_per_cycle=4)])
    cavities = []
    for i in range(4):
        result = aligner.align("cam", 104.0 + i * 0.1)
        assert result.cycle_id == "c1" and result.confidence > 0.5
        cavities.append(result.cavity)
        store.save_inspection(InspectionRecord("", "cam", 104.0 + i * 0.1, "fail" if i == 2 else "pass",
                                               cycle_id="c1", cavity=result.cavity))
    assert cavities == [0, 1, 2, 3]
    assert aligner.align("cam", 104.5).confidence < 0.5   # a fifth part from a 4-cavity shot is suspect
    parts = store.part_verdicts(0, 1e9)
    assert sorted(p["verdict"] for p in parts) == ["fail", "pass", "pass", "pass"]


def test_fifo_alignment_refuses_multi_cavity(tmp_path):
    from src.process.alignment import StationLink

    with pytest.raises(ValueError):
        StationLink.from_config("cam", {"mode": "fifo", "parts_per_cycle": 2})


def test_low_confidence_links_do_not_train_the_risk_model(tmp_path):
    from src.process.models import InspectionRecord

    store = _store_with_cycle(tmp_path)
    store.save_inspection(InspectionRecord("", "cam", 104.0, "fail", cycle_id="c1", alignment_confidence=0.2))
    assert store.labelled_dataset(min_confidence=0.0)[0]["label"] == 1
    assert store.labelled_dataset(min_confidence=0.5) == []


def test_failed_part_with_no_cycle_link_is_still_reported(tmp_path):
    from src.process.models import InspectionRecord

    store = _store_with_cycle(tmp_path)
    store.save_inspection(InspectionRecord("", "cam", 104.0, "fail", defect_type="flash"))
    parts = store.part_verdicts(0, 1e9)
    assert [p["verdict"] for p in parts] == ["fail"] and parts[0]["cycle_id"] is None


def test_stationary_part_is_one_part_until_it_leaves():
    from src.camera.part_inspection import PartPresenceTracker

    tracker = PartPresenceTracker("cam", max_visit_frames=300)
    frame = np.full((48, 64, 3), 90, np.uint8)
    closed = sum(tracker.update(frame, True, i / 30, i) is not None for i in range(3000))
    assert closed == 1
    for i in range(3000, 3004):
        tracker.update(frame, False, i / 30, i)
    assert sum(tracker.update(frame, True, i / 30, i) is not None for i in range(3004, 3320)) == 1


def test_source_failure_logs_its_reason_and_retries(caplog):
    from src.process.sources import CycleSource

    class Failing(CycleSource):
        name = "flaky"
        calls = 0

        def _run(self):
            type(self).calls += 1
            self._stop.set()
            raise ConnectionError("PLC unplugged")

    with caplog.at_level("ERROR"):
        Failing()._run_safe()
    assert "PLC unplugged" in caplog.text and "Reconnecting in 1s" in caplog.text


def test_pruned_curves_cannot_wipe_the_golden_reference(tmp_path):
    from src.process.features import GoldenCycleReference
    from src.process.service import ProcessIntelligenceService

    cfg = {"db_path": str(tmp_path / "p.db"), "models_dir": str(tmp_path / "m"), "curve_retention_days": 30,
           "model": {"golden_warmup_cycles": 10, "retrain_every_labels": 10**9}}
    service = ProcessIntelligenceService(cfg)
    sim = InjectionMoldingSimulator(seed=3, start_ts=1_000_000.0)
    cycles = [sim.next_cycle()[0] for _ in range(12)]
    for cycle in cycles:
        service.on_cycle(cycle)
    assert GoldenCycleReference.fit([type(c)(**{**c.__dict__, "curves": {}}) for c in cycles], min_cycles=10) is None

    golden = service.golden
    assert golden is not None and golden.curves
    service.store.prune_curves(0.0)
    service.retrain()
    assert service.golden is golden
    assert ProcessIntelligenceService(cfg).golden.curves


def test_old_curves_are_pruned_as_cycles_arrive(tmp_path, monkeypatch):
    from src.process import service as module

    monkeypatch.setattr(module, "PRUNE_EVERY_CYCLES", 5)
    service = module.ProcessIntelligenceService(
        {"db_path": str(tmp_path / "p.db"), "models_dir": str(tmp_path / "m"), "curve_retention_days": 1,
         "model": {"golden_warmup_cycles": 100}})
    sim = InjectionMoldingSimulator(seed=3, start_ts=1_000_000.0)
    cycles = [sim.next_cycle()[0] for _ in range(5)]
    for cycle in cycles:
        service.on_cycle(cycle)
    assert not service.store.get_cycle(cycles[0].cycle_id, with_curves=True).curves


def test_sensor_dropouts_do_not_blind_the_golden_reference():
    sim = InjectionMoldingSimulator(seed=4, start_ts=1_000_000.0)
    cycles = [sim.next_cycle()[0] for _ in range(12)]
    for cycle in cycles[::3]:
        cycle.curves = {name: np.full_like(values, np.nan) for name, values in cycle.curves.items()}
    golden = GoldenCycleReference.fit(cycles, min_cycles=5)
    assert golden.curves and all(np.isfinite(curve).all() for curve in golden.curves.values())


def test_part_nobody_could_judge_is_stored_but_never_a_training_label(tmp_path):
    from src.process.models import NOT_INSPECTED
    from src.process.service import ProcessIntelligenceService

    service = ProcessIntelligenceService({"db_path": str(tmp_path / "p.db"), "models_dir": str(tmp_path / "m"),
                                          "model": {"golden_warmup_cycles": 100}},
                                         cameras_cfg=[{"id": "cam", "process_link": {"machine_id": "imm_1",
                                                       "mode": "time", "transport_delay_s": 3.0, "tolerance_s": 2.0}}])
    sim = InjectionMoldingSimulator(seed=2, start_ts=1_000_000.0)
    cycles = [sim.next_cycle()[0] for _ in range(3)]
    for cycle in cycles:
        service.on_cycle(cycle)
    service.record_inspection("cam", cycles[0].end_ts + 3.0, NOT_INSPECTED)   # camera fault: used to raise
    service.record_inspection("cam", cycles[1].end_ts + 3.0, PASS)
    labelled = {r["cycle_id"] for r in service.store.labelled_dataset()}
    assert labelled == {cycles[1].cycle_id}
    verdicts = {p["cycle_id"]: p["verdict"] for p in service.store.part_verdicts(0, 2e9)}
    assert verdicts[cycles[0].cycle_id] == "not_inspected" and verdicts[cycles[1].cycle_id] == "pass"


def test_a_failed_view_still_beats_a_view_that_could_not_judge(tmp_path):
    from src.process.models import NOT_INSPECTED

    store = ProcessStore(tmp_path / "p.db")
    for camera, verdict in (("top", NOT_INSPECTED), ("side", FAIL), ("front", PASS)):
        store.save_inspection(InspectionRecord("", camera, 10.0, verdict, cycle_id="m-1"))
    assert store.part_verdicts(0, 100)[0]["verdict"] == "fail"
