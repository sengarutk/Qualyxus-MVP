"""Stage 2 of the audit: the numbers are true and the data behind them is kept."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time

import numpy as np
import pytest

from src.inference.audit_log import ChainVerifier, InspectionAuditLog, verify_chain, verify_incremental
from src.process.alignment import AlignmentResult
from src.process.kpis import compute_kpis
from src.process.models import FAIL, NOT_INSPECTED, PASS, InspectionRecord, ProcessCycle
from src.process.service import ProcessIntelligenceService
from src.process.simulator import InjectionMoldingSimulator
from src.process.sources import CycleAssembler, SignalSpec, build_cycle_sources
from src.process.store import MAINTENANCE, PROCESS_CHANGE, ProcessStore

VIEWS = ("top", "side", "front")


@pytest.fixture
def store(tmp_path):
    return ProcessStore(tmp_path / "p.db")


def _cycle(n, machine="m", ts=None, complete=True, **scalars):
    end = 100.0 + 10 * n if ts is None else ts
    return ProcessCycle(f"{machine}-{n}", machine, n, end - 5, end, scalars=scalars, complete=complete)


def _view(store, cycle_id, camera, verdict, ts=100.0, **kw):
    kw.setdefault("alignment_confidence", 0.9)
    store.save_inspection(InspectionRecord("", camera, ts, verdict, cycle_id=cycle_id, **kw))


# ------------------------------------------------------------------------------- KPIs (B1, B2)

def test_scrap_rate_counts_parts_and_ignores_what_was_not_judged(store):
    for i in range(10):
        for v in VIEWS:
            _view(store, f"m-{i}", v, FAIL if (i == 0 and v == "top") else PASS, 100 + i)
    for i in range(10, 14):
        for v in VIEWS:
            _view(store, f"m-{i}", v, NOT_INSPECTED if v == "side" else PASS, 100 + i)
    _view(store, "m-20", "top", PASS, 130)      # two views never reported
    k = compute_kpis(store, 0, 1e4, required_cameras=VIEWS)
    assert (k["parts_inspected"], k["parts_failed"], k["parts_not_inspected"]) == (10, 1, 5)
    assert k["scrap_rate"] == pytest.approx(0.1)


def test_unlinked_views_of_one_part_are_one_part_and_cannot_pass(store):
    for i in range(4):
        for j, v in enumerate(VIEWS):
            _view(store, None, v, FAIL if (i == 3 and v == "side") else PASS, 100 + 10 * i + 0.3 * j,
                  alignment_confidence=None)
    parts = store.part_verdicts(0, 1e4, required_cameras=VIEWS)
    assert [p["verdict"] for p in parts] == ["not_inspected"] * 3 + ["fail"]


def test_late_unlinked_view_is_not_joined_to_an_older_part(store):
    _view(store, None, "top", PASS, 100.0)
    _view(store, None, "side", FAIL, 130.0)
    assert len(store.part_verdicts(0, 1e4, required_cameras=VIEWS)) == 2


def test_single_camera_unlinked_parts_still_pass(store):
    _view(store, None, "only", PASS)
    assert store.part_verdicts(0, 1e4)[0]["verdict"] == "pass"


def test_audit_rates_use_the_whole_part_even_when_vision_was_earlier(store):
    for v in VIEWS:
        _view(store, "m-1", v, FAIL if v == "side" else PASS, 100.0)     # rejected; a person says good
        _view(store, "m-2", v, PASS, 101.0)                               # passed; a person says bad
        _view(store, "m-3", v, NOT_INSPECTED, 102.0)
    for c, verdict in (("m-1", PASS), ("m-2", FAIL), ("m-3", FAIL)):
        _view(store, c, "audit", verdict, 5000.0, source="audit")
    k = compute_kpis(store, 4000, 6000, required_cameras=VIEWS)
    assert k["false_reject_rate"] == 1.0 and k["escape_rate"] == 1.0     # m-3 was never passed: not an escape
    assert k["parts_inspected"] == 0 and k["scrap_rate"] is None


# ------------------------------------------------------------------------------ labels (B3, B4)

@pytest.mark.parametrize("views,label", [
    ({"top": PASS, "side": PASS, "front": PASS}, 0),
    ({"top": PASS, "side": NOT_INSPECTED, "front": PASS}, None),
    ({"top": PASS, "side": PASS}, None),
    ({"top": PASS, "side": NOT_INSPECTED, "front": FAIL}, 1),
])
def test_only_a_fully_inspected_part_is_a_good_label(store, views, label):
    store.save_cycle(_cycle(1), {"a": 1.0})
    for camera, verdict in views.items():
        _view(store, "m-1", camera, verdict)
    rows = store.labelled_dataset(required_cameras=VIEWS)
    assert [r["label"] for r in rows] == ([] if label is None else [label])
    assert bool(store.good_cycles(with_curves=False, min_views=3)) == (label == 0)


def test_a_failure_that_may_belong_elsewhere_leaves_no_good_label(store):
    store.save_cycle(_cycle(1), {"a": 1.0})
    _view(store, "m-1", "top", PASS)
    _view(store, "m-1", "side", FAIL, alignment_confidence=0.2)
    assert store.labelled_dataset(min_confidence=0.5) == []
    _view(store, "m-1", "audit", PASS, source="audit")
    assert store.labelled_dataset(min_confidence=0.5)[0]["label"] == 0


def test_training_takes_the_newest_cycles(store):
    for n in range(20):
        store.save_cycle(_cycle(n), {"n": float(n)})
        _view(store, f"m-{n}", "top", PASS)
    assert [r["features"]["n"] for r in store.labelled_dataset(5)] == [15, 16, 17, 18, 19]


def _service(tmp_path, **cfg):
    cams = [{"id": v, "required_view": True,
             "process_link": {"machine_id": "imm_1", "transport_delay_s": 0.0, "tolerance_s": 1.0}} for v in VIEWS]
    return ProcessIntelligenceService({"db_path": str(tmp_path / "p.db"), "models_dir": str(tmp_path / "m"),
                                       **cfg}, cameras_cfg=cams)


def test_one_part_is_one_label_and_a_late_reject_is_not_a_good_baseline(tmp_path):
    svc = _service(tmp_path)
    sim = InjectionMoldingSimulator(seed=3, start_ts=1e6)
    good, bad = sim.next_cycle()[0], sim.next_cycle()[0]
    for c in (good, bad):
        svc.on_cycle(c)
    for v in VIEWS:
        svc.record_inspection(v, good.end_ts, PASS, alignment=AlignmentResult(good.cycle_id, 0.9, "time"))
        svc.record_inspection(v, bad.end_ts, FAIL if v == "front" else PASS,
                              alignment=AlignmentResult(bad.cycle_id, 0.9, "time"))
    assert svc._labels_since_train == 2
    assert len(svc._state("imm_1").regime_good) == 1


# ---------------------------------------------------------------------- lost cycles (B6, B7)

def _feed(asm, t0, seconds, counter, hz=20):
    out = [asm.feed(t0 + i / hz, {"p": 1.0}, counter) for i in range(int(seconds * hz))]
    return [c for c in out if c]


def test_outage_inside_a_cycle_keeps_the_link_but_not_the_data():
    asm = CycleAssembler("m", [SignalSpec("p")])
    _feed(asm, 0, 3, 99)
    _feed(asm, 3, 1, 100)
    _feed(asm, 63, 2, 100)
    (cycle,) = _feed(asm, 65, 1, 101)
    assert cycle.sequence == 100 and not cycle.complete and cycle.duration_s < 3
    (after,) = _feed(asm, 66, 1, 102)
    assert after.sequence == 101 and after.complete


def test_cycle_that_ended_during_an_outage_is_not_given_an_invented_end():
    asm = CycleAssembler("m", [SignalSpec("p")])
    _feed(asm, 0, 3, 100)
    assert _feed(asm, 60, 2, 105) == []
    (cycle,) = _feed(asm, 62, 1, 106)
    assert cycle.sequence == 105 and not cycle.complete


def test_first_cycle_after_start_up_is_not_complete():
    asm = CycleAssembler("m", [SignalSpec("p")])
    _feed(asm, 0, 1, 7)
    cycles = _feed(asm, 1, 1, 8) + _feed(asm, 2, 1, 9)
    assert [c.complete for c in cycles] == [False, True]


def test_lost_cycles_are_recorded_also_across_a_restart_and_teach_nothing(tmp_path):
    svc = _service(tmp_path)
    svc.on_cycle(_cycle(10, "imm_1"))
    svc.on_cycle(_cycle(13, "imm_1"))
    again = _service(tmp_path)
    again.on_cycle(_cycle(20, "imm_1", complete=False))
    again.on_cycle(_cycle(1, "imm_1", ts=400.0))            # counter reset: nothing can be known
    gaps = again.store.gaps_between(0, 1e12)
    assert [(g["first_sequence"], g["last_sequence"], g["missed"]) for g in gaps] == [(11, 12, 2), (14, 19, 6)]
    assert compute_kpis(again.store, 0, 1e12)["machine_cycles_lost"] == 8
    assert len(again.store.cycle_features(limit=100, newest=True)) == 3      # the incomplete one is left out
    _view(again.store, "imm_1-20", "top", PASS)
    assert again.store.labelled_dataset() == [] and again.store.good_cycles() == []


# ------------------------------------------------------------- baselines per machine (B8, B9)

def test_machines_do_not_share_a_baseline_or_a_model(tmp_path):
    both = {"machines": [{"machine_id": "imm_1"}, {"machine_id": "imm_2"}], "model": {"golden_warmup_cycles": 30}}
    svc = _service(tmp_path, **both)
    a = InjectionMoldingSimulator("imm_1", seed=3, start_ts=1e6, disturbance_scale=0.0)
    b = InjectionMoldingSimulator("imm_2", seed=4, start_ts=1e6, disturbance_scale=0.0)
    b.apply_setpoints({"hold_pressure_sp": 450.0, "melt_temp_sp": 222.0})
    for _ in range(60):
        for sim in (a, b):
            svc.on_cycle(sim.next_cycle()[0])
    one, two = svc._state("imm_1"), svc._state("imm_2")
    assert one.model.version != two.model.version
    assert svc.store.active_model("imm_1")["version"] == one.model.version
    assert svc.store.active_model("imm_2")["version"] == two.model.version
    sp = "setpoint.hold_pressure_sp"
    rows = [{r[sp] for r in svc.store.cycle_features(limit=999, newest=True, machine_id=m)} for m in ("imm_1", "imm_2")]
    assert rows[1] == {450.0} and rows[0].isdisjoint(rows[1])
    again = _service(tmp_path, **both)
    assert again._state("imm_2").model.version == two.model.version
    assert again._state("imm_2").golden is not None


def test_config_builds_one_source_per_machine():
    sources = build_cycle_sources({"source": "simulated", "simulator": {"time_scale": 5.0}, "machines": [
        {"machine_id": "a"}, {"machine_id": "b", "simulator": {"seed": 1}}]})
    assert [s.simulator.machine_id for s in sources] == ["a", "b"] and sources[0].time_scale == 5.0
    assert len(build_cycle_sources({"source": "simulated"})) == 1


def test_machines_need_their_own_names():
    from src.config.schema import ConfigError, validate

    with pytest.raises(ConfigError, match="machine_id"):
        validate({"process": {"machines": [{"machine_id": "a"}, {"machine_id": "a"}]}})
    with pytest.raises(ConfigError, match="unknown key"):
        validate({"process": {"machines": [{"machine_id": "a", "sorce": "modbus"}]}})


def test_repair_starts_the_health_baseline_again(tmp_path):
    from src.process.baseline import main as baseline_cli
    from src.process.machine_health import assess_machine

    svc = _service(tmp_path)
    sim = InjectionMoldingSimulator("imm_1", seed=3, start_ts=time.time() - 9000, disturbance_scale=0.0)
    for i in range(160):
        if i == 120:
            sim.check_ring_wear = 0.2
        svc.on_cycle(sim.next_cycle()[0])

    def ring():
        return {c.component: c.status for c in assess_machine(*svc.store.health_rows("imm_1"))}.get("Check ring")
    assert ring() == "ACT"
    model = svc.model
    baseline_cli(["reset", "--machine", "imm_1", "--by", "asha", "--reason", "ring", "--db", str(tmp_path / "p.db")])
    sim.check_ring_wear = 0.0
    for i in range(40):
        c, _ = sim.next_cycle()
        c.start_ts, c.end_ts = time.time() + i, time.time() + i + 0.5
        svc.on_cycle(c)
    assert ring() == "OK" and svc.model is model       # a repair does not throw the learned model away
    with pytest.raises(SystemExit):
        baseline_cli(["reset", "--machine", "typo", "--by", "a", "--reason", "b", "--db", str(tmp_path / "p.db")])


def test_new_mould_starts_baseline_golden_and_model_again(tmp_path):
    cfg = {"baseline_reset_on": ["mould"], "model": {"golden_warmup_cycles": 20}}
    svc = _service(tmp_path, **cfg)
    sim = InjectionMoldingSimulator("imm_1", seed=3, start_ts=1e6, disturbance_scale=0.0)

    def run(n, mould):
        for _ in range(n):
            c, _ = sim.next_cycle()
            c.scalars["mould"] = mould
            svc.on_cycle(c)
    run(40, 11.0)
    first = svc.model.version
    svc = _service(tmp_path, **cfg)                      # the mould is changed while the software is off
    assert svc.model.version == first
    svc.store.reset_baseline("imm_2", reason="other machine", by="x")
    run(5, 12.0)
    assert svc.model is None and svc.golden is None
    assert len(svc.store.health_rows("imm_1")[0]) == 5
    run(30, 12.0)
    assert svc.model.version != first
    assert svc.store.baseline_start("imm_1", MAINTENANCE) == 0 < svc.store.baseline_start("imm_1", PROCESS_CHANGE)


def test_old_model_does_not_come_back_after_a_restart(tmp_path):
    svc = _service(tmp_path, model={"golden_warmup_cycles": 20})
    sim = InjectionMoldingSimulator("imm_1", seed=3, start_ts=time.time() - 9000, disturbance_scale=0.0)
    for _ in range(25):
        svc.on_cycle(sim.next_cycle()[0])
    assert svc.model is not None
    svc.store.reset_baseline("imm_1", kind=PROCESS_CHANGE, reason="new mould", by="asha", ts=time.time() + 1)
    again = _service(tmp_path)
    assert again.model is None and again.golden is None


# ----------------------------------------------------------------------- corrections (B24)

def test_worn_check_ring_gets_one_repair_request_and_no_pressure_rise(tmp_path):
    svc = _service(tmp_path, model={"golden_warmup_cycles": 100})
    sim = InjectionMoldingSimulator("imm_1", seed=3, start_ts=1e6, disturbance_scale=0.0)
    issued, high = [], []
    svc.add_alert_callback(lambda a, c, p: (issued.extend(p), high.append(a.decision)))
    for i in range(220):
        if i == 120:
            sim.check_ring_wear = 0.2
        svc.on_cycle(sim.next_cycle()[0])
    assert "high" in high
    open_now = [a["action_payload"] for a in svc.tracker.list_actions(status="open", limit=999)]
    assert [p["component"] for p in open_now if p["action_type"] == "process_maintenance"] == ["Check ring"]
    assert not [p for p in open_now if p.get("setpoint") == "hold_pressure_sp"]
    assert len([p for p in issued if p.setpoint == "hold_pressure_sp"]) <= 1     # before the wear could be known


def test_open_suggestions_are_withdrawn_but_never_a_started_one(tmp_path):
    from src.operations.action_tracker import ActionTracker
    from src.process.prescriptions import Prescription, PrescriptionEngine, withdraw

    tracker = ActionTracker(sqlite_path=str(tmp_path / "a.db"))

    def issue(sp):
        return PrescriptionEngine.issue(tracker, Prescription("imm_1", "c", sp, 1, 2, "f", 1, 1, .9, .1, "x"))["action_id"]
    stale, started, other = issue("melt_temp_sp"), issue("melt_temp_sp"), issue("mold_temp_sp")
    tracker.update_action(started, status="in_progress", changed_by="asha")
    assert withdraw(tracker, "imm_1", "replaced", setpoints=["melt_temp_sp"]) == 1
    state = {i: tracker.get_action(i) for i in (stale, started, other)}
    assert state[stale]["status"] == "closed" and state[stale]["verification_status"] == "WITHDRAWN"
    assert state[started]["status"] == "in_progress" and state[other]["status"] == "open"
    assert withdraw(tracker, "imm_2", "other machine") == 0


# ------------------------------------------------------------------------ models (B11 to B13)

def test_blind_model_raises_an_alarm_and_no_retraining():
    from src.inference.anomaly.monitoring import ModelHealthMonitor

    m = ModelHealthMonitor(calibration={"holdout_mean": 0.5}, threshold=1.0)
    for _ in range(40):
        status = m.update(0.05, False)
    assert "blind" in status["alerts"][0] and status["score_drift"] < m.drift_ratio
    ok = ModelHealthMonitor(calibration={"holdout_mean": 0.5}, threshold=1.0)
    for _ in range(40):
        status = ok.update(0.45, False)
    assert not status["alerts"]


def _registry(tmp_path, n=3):
    from src.inference.model_registry import VisionModelRegistry

    reg = VisionModelRegistry(tmp_path / "reg")
    for i in range(n):
        (tmp_path / f"m{i}.pt").write_text(str(i))
        (tmp_path / f"m{i}.json").write_text('{"threshold": 1}')
        reg.register("a", "anomaly", tmp_path / f"m{i}.pt", activate=True)
    return reg


def test_rollback_walks_back_and_never_returns_to_a_bad_model(tmp_path):
    reg = _registry(tmp_path)

    def back():
        return reg.rollback("a", "anomaly", by="x", reason="bad")
    assert [back(), back()] == ["v0002", "v0001"]
    with pytest.raises(ValueError):
        back()
    (tmp_path / "m9.pt").write_text("9")
    reg.register("a", "anomaly", tmp_path / "m9.pt", activate=True)
    assert back() == "v0001"                    # not v0002 or v0003


@pytest.mark.parametrize("name", ["m2.pt", "m2.json"])
def test_changed_model_file_is_refused_and_parts_are_not_inspected(tmp_path, name, monkeypatch):
    import src.inference.inspection_stack as stack_module
    from src.inference.inspection_stack import StackProvider

    reg = _registry(tmp_path)
    loaded = []
    monkeypatch.setattr(stack_module, "_load_anomaly", lambda p: loaded.append(p) or (object() if p else None))
    assert not StackProvider({"registry_dir": str(reg.root), "recipe": "a"}).get()[0].empty
    (reg.root / "a" / "anomaly" / "v0003" / name).write_text("tampered")
    assert reg.verify("a", "anomaly", "v0003")
    stack = StackProvider({"registry_dir": str(reg.root), "recipe": "a"}).get()[0]
    assert stack.empty and loaded[-1] is None   # the file is never opened as a model


def test_models_registered_before_full_digests_still_load(tmp_path):
    reg = _registry(tmp_path, 1)
    path = reg.root / "a" / "anomaly" / "manifest.json"
    m = json.loads(path.read_text())
    del m["versions"]["v0001"]["sha256"]
    path.write_text(json.dumps(m))
    assert reg.verify("a", "anomaly", "v0001") is None
    (reg.root / "a" / "anomaly" / "v0001" / "m0.pt").write_text("x")
    assert reg.verify("a", "anomaly", "v0001")


# --------------------------------------------------------------------------- audit check (B18)

@pytest.fixture
def days(tmp_path, monkeypatch):
    import src.inference.audit_log as mod

    log = InspectionAuditLog(tmp_path / "audit", tmp_path / "k")
    with monkeypatch.context() as m:
        for d in range(4):
            m.setattr(mod.time, "strftime", lambda fmt, *a, d=d: f"2026-01-{d + 1:02d}")
            for i in range(5):
                log.append({"part": i})
    return tmp_path / "audit", tmp_path / "k", sorted((tmp_path / "audit").glob("*.jsonl")), log


def test_repeat_checks_read_only_the_newest_day(days):
    directory, key, _, log = days
    v = ChainVerifier(directory, key)
    assert v.check() == verify_chain(directory, key) and v.files_read == 4
    log.append({"part": 99})
    result = v.check()
    assert result["ok"] and result["records"] == 21 and v.files_read <= 2
    assert v.check()["ok"] and v.files_read == 1


def test_an_old_day_changed_later_is_found_even_with_its_time_put_back(days):
    directory, key, files, _ = days
    v = ChainVerifier(directory, key)
    assert v.check()["ok"]
    before = files[1].stat()
    files[1].write_text(files[1].read_text().replace('"part": 3', '"part": 4'))
    os.utime(files[1], ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not v.check()["ok"]
    assert not v.check()["ok"]          # and it stays found


def test_deleted_day_and_changed_key_are_found_by_the_quick_check(days):
    directory, key, files, _ = days
    assert verify_incremental(directory, key)["ok"]
    body = files[2].read_text()
    files[2].unlink()
    assert not verify_incremental(directory, key)["ok"]
    files[2].write_text(body)
    assert verify_incremental(directory, key)["ok"]
    key.write_text("00" * 32)
    assert not verify_incremental(directory, key)["ok"]


def test_restart_continues_the_count(days):
    directory, key, _, _ = days
    log = InspectionAuditLog(directory, key)
    log.append({"part": 1})
    assert log._records == 21 and verify_chain(directory, key)["ok"]


def test_console_says_checking_not_intact_while_a_long_check_runs(days, monkeypatch):
    import src.console.data as console
    import src.inference.audit_log as mod

    directory, key, _, _ = days
    data = console.ConsoleData({"inspection": {"audit": {"dir": str(directory), "key_file": str(key)}}})
    monkeypatch.setattr(console, "VERIFY_WAIT_S", 0.1)
    real = mod.verify_incremental
    monkeypatch.setattr(mod, "verify_incremental", lambda *a: time.sleep(0.6) or real(*a))
    first = data.verify()
    assert first["ok"] is None and first["checking"]
    data._verifying.join(5)
    assert data._audit_cache[1]["ok"] is True
    monkeypatch.setattr(mod, "verify_incremental", lambda *a: 1 / 0)
    data._audit_cache = (0.0, {})
    assert data.verify()["ok"] is False


# ----------------------------------------------------------------------- event writer (B19, B20)

@pytest.fixture
def events(tmp_path):
    from src.logging.event_logger import EventLogger

    before, EventLogger._instance = EventLogger._instance, None
    log = EventLogger(str(tmp_path / "e.db"), str(tmp_path / "e.csv"))
    yield log
    EventLogger._instance = before


def _event(n):
    return {"camera_id": "c", "box": [1], "track_id": n, "timestamp": 1.0}


def _saved(log):
    with sqlite3.connect(log.db_path) as conn:
        return [r[0] for r in conn.execute("SELECT track_id FROM events ORDER BY track_id")]


def _run(log, items, seconds, timeout_sec=0.05, late=()):
    async def go():
        log.queue = asyncio.Queue()
        log.timeout_sec = timeout_sec
        await log.start()
        for item in items:
            log.queue.put_nowait(item)
        await asyncio.sleep(seconds)
        assert not log.worker_task.done()
        for item in late:
            log.queue.put_nowait(item)
        log.worker_task.cancel()
        await asyncio.gather(log.worker_task, return_exceptions=True)
    asyncio.run(go())


def test_event_writer_outlives_a_locked_database(events):
    real, calls = events._sync_execute_transaction, [0]

    def flaky(batch, keep_on_failure=True):
        calls[0] += 1
        if calls[0] <= 2:
            raise sqlite3.OperationalError("database is locked")
        return real(batch, keep_on_failure)
    events._sync_execute_transaction = flaky
    _run(events, [_event(n) for n in range(5)], 1.5)
    assert _saved(events) == [0, 1, 2, 3, 4] and calls[0] == 3


def test_one_bad_record_does_not_cost_the_others(events):
    bad = {"event_id": "e", "source": "c", "full_image_path": "", "defect_type": "x", "timestamp": 1.0,
           "confidence": None, "severity": 1}       # the column is NOT NULL
    _run(events, [_event(0), bad, _event(1)], 0.8)
    assert _saved(events) == [0, 1]
    assert len(open(events.unsaved_path).read().splitlines()) == 1


@pytest.mark.parametrize("moment", ["collecting", "writing", "retrying"])
def test_shutdown_loses_no_event_and_writes_none_twice(events, moment):
    real, calls = events._sync_execute_transaction, [0]

    def slow(batch, keep_on_failure=True):
        calls[0] += 1
        if moment == "retrying" and calls[0] == 1:
            raise sqlite3.OperationalError("database is locked")
        if moment == "writing":
            time.sleep(0.4)
        return real(batch, keep_on_failure)
    events._sync_execute_transaction = slow
    _run(events, [_event(n) for n in range(3)], 0.15, 5 if moment == "collecting" else 0.02, late=[_event(3)])
    assert _saved(events) == [0, 1, 2, 3]


def test_events_go_to_a_file_when_the_database_never_returns(events):
    def down(batch, keep_on_failure=True):
        raise sqlite3.OperationalError("disk I/O error")
    events._sync_execute_transaction = down
    _run(events, [_event(1)], 0.2)
    assert json.loads(open(events.unsaved_path).read())["track_id"] == 1


# ------------------------------------------------------------- nothing invented (B21, B22), B23

@pytest.mark.parametrize("mode", ["simulated", "plc"])
def test_telemetry_is_empty_until_something_is_measured(mode):
    from src.telemetry.telemetry_provider import TelemetryProvider

    ctx = TelemetryProvider(mode=mode, endpoint="127.0.0.1").get_telemetry("cam_0")
    for name in ("machine_id", "line_id", "station_id", "temperature", "vibration", "speed", "lot_id", "shift_id"):
        assert getattr(ctx, name) is None, name


class _Cfg:
    def __init__(self, **vlm):
        self.raw = {"vlm_reasoning": vlm}


def test_language_model_errors_never_become_a_root_cause(monkeypatch):
    import httpx

    import src.inference.analysis.vlm_gateway as gw

    crop = np.zeros((8, 8, 3), np.uint8)
    called = []

    class Client:
        reply = (500, {})

        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *_): return False

        def post(self, url, json):
            called.append(url)
            return httpx.Response(self.reply[0], json=self.reply[1], request=httpx.Request("POST", url))
    monkeypatch.setattr(gw.httpx, "Client", Client)
    assert gw.analyze_defect_crop_sync(crop, "x", _Cfg()) == "" and not called      # off unless asked for
    for Client.reply in ((500, {}), (200, {}), (200, {"response": ""}), (404, {"response": "x"})):
        assert gw.analyze_defect_crop_sync(crop, "x", _Cfg(enabled=True)) == ""
    Client.reply = (200, {"response": "Short shot."})
    assert gw.analyze_defect_crop_sync(crop, "x", _Cfg(enabled=True)) == "AI suggestion, not verified: Short shot."


def test_one_hot_reading_stops_nothing_and_a_real_emergency_ends_when_cool(monkeypatch):
    import src.system.resource_monitor as rm
    from src.config.config import Config
    from src.logging.schema import SystemState, SystemVitals

    now = [1000.0]
    gov = rm.ResourceGovernor(Config.load(overrides={"resource_management": {"emergency_hold_seconds": 5,
                                                                             "emergency_cooldown_seconds": 60}}))

    class Clock:
        time = staticmethod(lambda: now[0])
    monkeypatch.setattr(rm, "time", Clock)
    temp = [40.0]
    gov.get_vitals = lambda: SystemVitals(cpu_load_percent=1, ram_percent=1, cpu_temperature_c=temp[0],
                                          gpu_temperature_c=0, vram_percent=1, timestamp=now[0])
    called = []
    gov.set_emergency_callback(lambda v: called.append(v))

    def at(seconds, t):
        now[0], temp[0] = now[0] + seconds, t
        return gov.get_policy_state()
    assert at(0, 40) == SystemState.SAFE
    assert at(1, 95) == SystemState.SAFE and at(1, 40) == SystemState.SAFE       # a glitch
    at(1, 95)
    assert at(6, 95) == SystemState.EMERGENCY and len(called) == 1
    assert at(30, 60) == SystemState.EMERGENCY
    assert at(20, 88) == SystemState.EMERGENCY            # hot again: the wait starts over
    assert at(30, 60) == SystemState.EMERGENCY and at(40, 60) == SystemState.EMERGENCY
    assert at(30, 60) == SystemState.WARNING
    at(1, 95)
    assert at(6, 95) == SystemState.EMERGENCY and len(called) == 2


# ------------------------------------------------------------------------------- upgrades

def test_database_from_before_this_version_is_upgraded_in_place(tmp_path):
    path = tmp_path / "old.db"
    store = ProcessStore(path)
    store.save_cycle(_cycle(1), {"a": 1.0})
    store.close()
    with sqlite3.connect(path) as conn:
        conn.execute("ALTER TABLE process_cycles DROP COLUMN complete")
        conn.execute("ALTER TABLE model_registry DROP COLUMN machine_id")
        conn.execute("DROP TABLE cycle_gaps")
        conn.execute("INSERT INTO model_registry VALUES ('v', 1.0, 'deviation', 'x', 1, 0, '{}', 1)")
    new = ProcessStore(path)
    assert new.get_cycle("m-1").complete and new.gaps_between(0, 1e12) == []
    assert new.active_model("imm_1")["version"] == "v"
    assert new.active_model("imm_2", any_machine=False) is None
