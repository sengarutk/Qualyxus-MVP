"""Stage 3 of the audit: ready for a real camera, a real PLC and months of running."""
from __future__ import annotations

import socket
import threading
import time

import cv2
import numpy as np
import pytest

from src.camera.camera_manager import CameraManager, pool_slots
from src.camera.part_inspection import CompletedPart, PartPresenceTracker, PartVerdictFuser
from src.camera.presence import PresenceDetector, is_part_present, learn_reference
from src.camera.trigger import PartTrigger, TriggerSource
from src.inference.anomaly.image_quality import image_fault, image_quality_baseline
from src.process.alignment import CycleAligner, StationLink
from src.process.features import curve_statistics
from src.process.models import FAIL, PASS, InspectionRecord, ProcessCycle
from src.process.sources import ModbusCycleSource, OpcUaCycleSource, SignalSpec
from src.process.store import ProcessStore
from src.process.synthetic_parts import FACES, render_part
from src.telemetry.plc_adapter import PLCAdapterConfig
from test_model_ops import _StubStack, _health_fields, _reject_path_manager

H, W = 480, 640
BLANK = np.full((H, W, 3), 40, np.uint8)
GRADIENT = cv2.merge([np.tile(np.linspace(60, 110, W, dtype=np.uint8), (H, 1))] * 3)


def _noisy(img, sigma=2.0, seed=0):
    return np.clip(img + np.random.default_rng(seed).normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def _with_part(bg, level=170):
    img = bg.copy()
    cv2.rectangle(img, (200, 150), (440, 330), (level,) * 3, -1)
    return img


# ---------------------------------------------------------------------------- presence (A1)

@pytest.mark.parametrize("bg", [BLANK, GRADIENT, np.clip(GRADIENT.astype(int) + GRADIENT.transpose(0, 1, 2)[::-1] // 3, 0, 255).astype(np.uint8)])
def test_parts_separate_on_plain_and_uneven_light(bg):
    tracker, parts = PartPresenceTracker("c"), 0
    for k in range(10):
        for j in range(16):
            frame = _noisy(_with_part(bg) if j >= 8 else bg, seed=100 * k + j)
            parts += tracker.update(frame, is_part_present(frame), k + j / 20, j) is not None
    assert parts >= 9


def test_low_contrast_part_is_seen_and_noise_is_not():
    assert is_part_present(_noisy(_with_part(BLANK, 47)))
    assert not any(is_part_present(_noisy(BLANK, 4.0, s)) for s in range(20))


@pytest.mark.parametrize("face", range(6))
@pytest.mark.parametrize("defect", [None, "short_shot"])
def test_every_face_is_present(face, defect):
    assert is_part_present(render_part(W, H, defect, 3, face=face))


def test_patterned_station_needs_and_uses_a_reference():
    tile = np.indices((H, W)).sum(0) // 20 % 2 * 60 + 50
    bg = cv2.merge([tile.astype(np.uint8)] * 3)
    detector = PresenceDetector(reference=learn_reference([_noisy(bg, seed=s) for s in range(7)]))
    assert not any(detector(_noisy(bg, seed=50 + s)) for s in range(5))
    assert detector(_noisy(_with_part(bg, 200)))
    assert not detector(np.clip(_noisy(bg) * 0.85, 0, 255).astype(np.uint8))     # the lamp dims: not a part


def test_part_filling_the_frame_is_seen_with_a_reference():
    assert PresenceDetector(reference=BLANK)(np.full_like(BLANK, 150))


def test_reference_follows_slow_light_but_never_absorbs_a_part():
    detector = PresenceDetector(reference=BLANK)
    part = _with_part(BLANK, 52)
    for _ in range(400):
        assert detector(part)


def test_reference_learning_refuses_a_moving_scene():
    with pytest.raises(ValueError, match="not empty"):
        learn_reference([_with_part(BLANK) if s % 3 == 0 else BLANK for s in range(9)])


def test_missing_reference_stops_the_start():
    with pytest.raises(ValueError, match="reference"):
        PresenceDetector.from_config({"reference": "/nope.png"})


# ------------------------------------------------------------------------------ trigger (A1)

def _trigger_manager(tmp_path, cameras=("top", "side")):
    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v", 0.4))
    _health_fields(cm, cameras)
    cm.part_fuser = PartVerdictFuser(cameras, timeout_s=5)
    cm.trigger = PartTrigger(list(cameras), frames=2, delay_s=0.0, timeout_s=5.0)
    cm.trigger_source = type("S", (), {"alive": lambda self: True, "stop": lambda self: None})()
    cm.inspection_enabled, cm.part_trackers = True, {}
    cm._presence_on_frames = cm._presence_off_frames = 1
    cm.inspection_executor = type("Now", (), {"submit": staticmethod(lambda f, *a: f(*a))})()
    return cm


def _verdicts(cm):
    out = []
    while not cm.fast_queue.empty():
        out.append(cm.fast_queue.get())
    return out


def test_triggered_part_is_judged_without_any_presence_check(tmp_path):
    cm = _trigger_manager(tmp_path)
    for number in (7, 8):
        cm._on_trigger(number, 0)
        for camera in ("top", "side"):
            for k in range(3):
                cm._track_part(cm.workers[camera], BLANK, k, [], [])       # an empty-looking frame
    verdicts = _verdicts(cm)
    assert [(v["verdict"], v["part"]) for v in verdicts] == [("pass", 7), ("pass", 8)]


def test_trigger_without_an_image_or_lost_is_not_inspected(tmp_path):
    cm = _trigger_manager(tmp_path)
    cm._on_trigger(1, 0)
    for k in range(2):
        cm._track_part(cm.workers["top"], BLANK, k, [], [])
    cm._on_trigger(3, 1)        # trigger 2 was never read; side never saw part 1
    verdicts = _verdicts(cm)
    assert [v["verdict"] for v in verdicts] == ["not_inspected", "not_inspected"]
    assert verdicts[1]["part"] == 1
    cm._report_missed(cm.trigger.expire(float("inf")))
    assert [(v["verdict"], v["part"]) for v in _verdicts(cm)] == [("not_inspected", 3)]


def test_unreadable_trigger_stops_the_heartbeat(tmp_path):
    cm = _trigger_manager(tmp_path)
    assert cm._heartbeat_hold(time.monotonic()) is None
    cm.trigger_source = type("S", (), {"alive": lambda self: False})()
    assert "trigger" in cm._heartbeat_hold(time.monotonic())


def test_frames_from_before_the_trigger_are_not_used():
    t = PartTrigger(["c"], frames=1, delay_s=0.1)
    t.fire("k", 10.0)
    assert t.frame("c", BLANK, 1, 10.05, [], []) is None
    part = t.frame("c", BLANK, 2, 10.2, [], [])
    assert part.trigger_key == "k" and part.ts == 10.0 and part.best_frame_id == 2


def test_trigger_counter_wraps_resets_and_reports_lost_ones():
    seen = []
    source = TriggerSource(PLCAdapterConfig(mode="plc"), lambda n, lost: seen.append((n, lost)))
    values = iter([65534, 65534, 65535, 0, 3, None, 4, 60000])
    source.adapter._read = lambda name: next(values)
    for _ in range(8):
        source.poll()
    assert seen == [(65535, 0), (0, 0), (3, 2), (4, 0), (60000, 0)]


def test_trigger_needs_a_plc():
    from src.config.schema import ConfigError, validate

    with pytest.raises(ConfigError, match="trigger"):
        validate({"inspection": {"trigger": {"enabled": True}}})


# ------------------------------------------------------------------------- linking views (A8)

def _aligner(tmp_path, ends, views=("a", "b", "c"), delays=None, cavities=1):
    store = ProcessStore(tmp_path / "p.db")
    for n, end in enumerate(ends, 1):
        store.save_cycle(ProcessCycle(f"m-{n}", "m", n, end - 2, end), {})
    delays = delays or {}
    links = [StationLink(v, "m", transport_delay_s=delays.get(v, 4.0), tolerance_s=2.0, parts_per_cycle=cavities) for v in views]
    return CycleAligner(store, links), store


def test_views_of_one_part_never_split_between_cycles(tmp_path):
    aligner, store = _aligner(tmp_path, [100.0, 103.0, 106.0])
    for left in (101.4, 104.4):     # as far from one cycle as from the next
        keys = []
        for v, offset in (("a", 0.0), ("b", 0.2), ("c", 0.35)):
            result = aligner.align_view(v, left + 4.0 + offset)
            keys.append(result.cycle_id)
            store.save_inspection(InspectionRecord("", v, left, PASS, cycle_id=result.cycle_id))
        assert len(set(keys)) == 1
    counts = [store.link_counts(v, ["m-1", "m-2", "m-3"]) for v in ("a", "b", "c")]
    assert counts[0] == counts[1] == counts[2] and sorted(counts[0].values()) == [1, 1]


def test_cameras_further_down_the_line_still_group(tmp_path):
    aligner, _ = _aligner(tmp_path, [100.0, 103.0], delays={"b": 9.0})
    assert aligner.align_view("a", 104.1).cycle_id == aligner.align_view("b", 109.3).cycle_id == "m-1"


def test_a_missed_view_does_not_drag_in_the_next_part(tmp_path):
    aligner, store = _aligner(tmp_path, [100.0, 103.0, 106.0])
    for v in ("a", "b"):
        r = aligner.align_view(v, 104.0)
        store.save_inspection(InspectionRecord("", v, 104.0, PASS, cycle_id=r.cycle_id))
    assert [aligner.align_view(v, 107.0 + 0.1 * i).cycle_id for i, v in enumerate(("c", "a", "b"))] == ["m-2"] * 3


def test_fast_machine_is_not_grouped_across_parts(tmp_path):
    aligner, store = _aligner(tmp_path, [100.0, 100.8, 101.6])
    first = aligner.align_view("a", 104.0)
    second = aligner.align_view("b", 104.8)       # within a second, but another part
    assert (first.cycle_id, second.cycle_id) == ("m-1", "m-2")


# -------------------------------------------------------------------------- image faults (B10)

@pytest.fixture(scope="module")
def baseline():
    return image_quality_baseline([render_part(W, H, None, s) for s in range(40)])


def _shift(f, dx):
    return cv2.warpAffine(f, np.float32([[1, 0, dx], [0, 1, 0]]), (W, H), borderValue=(40,) * 3)


def _belt(f):
    wave = 12 * np.sin(np.arange(W) / 3)[None, :, None] + 12 * np.sin(np.arange(H) / 3)[:, None, None]
    return np.where((f.max(2) < 80)[..., None], f + wave, f)


CONDITIONS = {
    "darker": lambda f: f * 0.8, "brighter": lambda f: f * 1.4,
    "uneven light": lambda f: f * np.linspace(0.7, 1.2, W)[None, :, None],
    "half glare": lambda f: np.hstack([f[:, :320], np.full_like(f[:, 320:], 255)]),
    "half covered": lambda f: np.hstack([f[:, :320], np.zeros_like(f[:, 320:])]),
    "part out of frame": lambda f: _shift(f, 250), "belt pattern": _belt,
}


@pytest.mark.parametrize("name", CONDITIONS)
def test_imaging_fault_is_not_a_verdict(baseline, name):
    for s in range(10):
        bad = np.clip(CONDITIONS[name](render_part(W, H, None, 500 + s)), 0, 255).astype(np.uint8)
        assert image_fault(bad, baseline), name


def test_normal_parts_and_defects_are_not_called_camera_faults(baseline):
    for s in range(60):
        assert image_fault(render_part(W, H, None, 900 + s), baseline) is None
        assert image_fault(_shift(render_part(W, H, None, 900 + s), 40), baseline) is None
    for d in ("short_shot", "flash", "sink_mark", "burn_mark"):
        assert not any(image_fault(render_part(W, H, d, 900 + s), baseline) for s in range(15))


def test_strong_compression_is_a_fault(baseline):
    jpeg = lambda f: cv2.imdecode(cv2.imencode(".jpg", f, [1, 20])[1], 1)  # noqa: E731
    assert sum(bool(image_fault(jpeg(render_part(W, H, None, s)), baseline)) for s in range(10)) >= 9


def test_models_calibrated_before_still_get_the_old_checks():
    old = {"sharpness": 90.0, "noise": 1.4}
    assert image_fault(CONDITIONS["darker"](render_part(W, H, None, 1)).astype(np.uint8), old) is None
    assert image_fault(cv2.GaussianBlur(render_part(W, H, None, 1), (0, 0), 4), old)


# ------------------------------------------------------------ twelve megapixels (B14, B15, B16)

def test_frame_memory_is_a_budget():
    for h, w in ((480, 640), (3000, 4000), (6000, 8000)):
        slots = pool_slots(h, w)
        assert slots >= 4 and (slots * h * w * 3 <= 200 * 2 ** 20 or slots == 4)


def test_presence_and_sharpness_keep_up_at_12_mp():
    big = cv2.resize(_with_part(GRADIENT), (4000, 3000))
    tracker, detector = PartPresenceTracker("c"), PresenceDetector()
    tracker.update(big, detector(big), 0.0, 0)
    started = time.perf_counter()
    for i in range(10):
        tracker.update(big, detector(big), i * 0.1, i)
    assert (time.perf_counter() - started) / 10 < 0.1      # was about 200 ms


def test_part_is_found_where_it_is_in_a_large_frame():
    from src.inference.anomaly.preprocess import find_part

    small = render_part(1000, 750, None, 4)
    (cx, cy), (w, h), _ = find_part(small)
    (bx, by), (bw, bh), _ = find_part(cv2.resize(small, (4000, 3000)))
    assert abs(bx / 4 - cx) < 3 and abs(by / 4 - cy) < 3 and abs(max(bw, bh) / 4 - max(w, h)) < 6


def test_pool_and_review_obey_a_size_limit(tmp_path):
    from src.inference.anomaly.monitoring import ReviewQueue
    from src.inference.good_pool import GoodPartPool

    noise = np.random.default_rng(0).integers(0, 255, (200, 300, 3), np.uint8)     # about 180 kB each
    pool = GoodPartPool(tmp_path / "pool", "a", every_n=1, max_gb=0.002)
    for _ in range(12):
        pool.offer(noise, verdict="pass", normalized_score=0.1, camera_id="c")
    assert 1 <= len(pool) <= 5
    newest = pool.entries()[-1]
    pool.offer(noise, verdict="pass", normalized_score=0.1, camera_id="c")
    assert newest in pool.entries()                    # the oldest go first
    review = ReviewQueue(tmp_path / "review", max_gb=0.0005)
    kept = [review.add("c", noise, normalized_score=1.0, verdict="pass") for _ in range(8)]
    assert kept[0] is not None and kept[-1] is None and sum(k is not None for k in kept) <= 4


def test_full_disk_stops_image_collection_but_not_the_record(tmp_path):
    from src.inference.anomaly.monitoring import ReviewQueue

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v", 1.0))
    cm.review_queue = ReviewQueue(tmp_path / "review")
    cm.disk_monitor = type("D", (), {"allow_defect_save": lambda self: False})()
    part = CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, 3), 7, 3)
    cm._inspect_part(state, part)
    assert not (tmp_path / "review").exists() and state.parts_inspected == 1
    cm.disk_monitor = type("D", (), {"allow_defect_save": lambda self: True})()
    cm._inspect_part(state, part)
    assert list((tmp_path / "review").rglob("*.json"))


def test_backlog_is_limited_by_memory(tmp_path, monkeypatch):
    import src.camera.inspection as module

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v", 0.4))
    cm._inspection_slots = threading.Semaphore(64)
    cm.inspection_executor = type("Busy", (), {"submit": staticmethod(lambda *a: None)})
    frame = render_part(160, 120, None, 3)
    monkeypatch.setattr(module, "BACKLOG_BYTES", int(2.5 * frame.nbytes))
    for i in range(3):
        cm._queue_inspection(state, CompletedPart("cam", 1.0, 2.0, frame, i, 3))
    assert len(cm._pending_parts) == 2 and state.parts_not_inspected == 1


# ------------------------------------------------------------------------- batching, views (B30, C1)

@pytest.fixture(scope="module")
def face_models():
    from src.inference.anomaly.patchcore import PatchCoreDetector

    out = []
    for face in range(3):
        det = PatchCoreDetector(backbone="resnet18", image_size=160, device="cpu")
        det.configure_preprocessing(register=face == 1)
        det.fit([render_part(320, 240, None, 100 + i, face=face) for i in range(24)])
        out.append(det)
    return out


def test_batched_views_get_the_same_result_as_single_ones(face_models):
    from src.inference.anomaly.base import predict_many

    assert len({id(d._ensure_backbone()) for d in face_models}) == 1
    frames = [render_part(320, 240, d, 900 + i, face=i) for i, d in enumerate((None, "short_shot", "flash"))]
    for single, batched in zip([d.predict(f) for d, f in zip(face_models, frames)],
                               predict_many(list(zip(face_models, frames)))):
        assert batched.is_anomalous == single.is_anomalous
        assert batched.score == pytest.approx(single.score, rel=1e-3)
        assert np.allclose(batched.anomaly_map, single.anomaly_map, atol=1e-2 * max(1.0, single.score))


def test_each_face_has_its_own_model_and_rejects_the_others(face_models):
    for i, det in enumerate(face_models):
        for j in range(3):
            score = [det.predict(render_part(320, 240, None, 700 + s, face=j)).normalized_score for s in range(3)]
            assert (min(score) > 1.5) if i != j else (np.median(score) < 1.0)


def test_the_six_faces_differ():
    gray = [cv2.resize(cv2.cvtColor(render_part(W, H, None, 5, face=f), cv2.COLOR_BGR2GRAY), (64, 48)).astype(float).ravel()
            for f in range(len(FACES))]
    for i in range(6):
        for j in range(i):
            assert np.corrcoef(gray[i], gray[j])[0, 1] < 0.85


def test_camera_with_a_view_uses_that_views_model(tmp_path):
    cm, _ = _reject_path_manager(tmp_path, _StubStack(False, "v", 0.4))
    cm._model_views = {"top": "top"}
    assert cm._model_recipe("top", "housing") == "housing-top"
    assert cm._model_recipe("other", "housing") == "housing"


def test_waiting_views_are_judged_together_and_one_failure_costs_only_itself(tmp_path):
    cm, _ = _reject_path_manager(tmp_path, _StubStack(False, "v", 0.4))
    _health_fields(cm, ("a", "b", "c"))
    cm._inspection_slots = threading.Semaphore(64)
    cm.inspection_executor = type("Later", (), {"submit": staticmethod(lambda *a: None)})
    frame = render_part(160, 120, None, 3)
    for camera in ("a", "b", "a", "c"):
        cm._queue_inspection(cm.workers[camera], CompletedPart(camera, 1.0, 2.0, frame, 1, 3))
    seen, sizes = [], []
    cm._judge_together = lambda batch: sizes.append(len(batch)) or {}

    def inspect(state, part, *judged):
        seen.append(state.camera_id)
        if state.camera_id == "b":
            raise RuntimeError("boom")
    cm._inspect_part = inspect
    while cm._pending_parts:
        cm._inspect_pending()
    assert seen == ["a", "b", "a", "c"] and sizes == [2, 2] and cm._backlog_bytes == 0


def test_batched_result_of_another_model_is_not_used(tmp_path):
    stack = _StubStack(False, "v", 0.4)
    used = []
    judge = stack.judge
    stack.judge = lambda frame, **kw: used.append(kw.get("anomaly")) or judge(frame)
    cm, state = _reject_path_manager(tmp_path, stack)
    part = CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, 3), 7, 3)
    cm._inspect_part(state, part, (object(), "stale"))
    cm._inspect_part(state, part, (stack.anomaly, "fresh"))
    assert used == [None, "fresh"]


# ----------------------------------------------------------------------------- retention (B17)

def test_old_rows_go_and_the_baseline_stays(tmp_path):
    store = ProcessStore(tmp_path / "p.db")
    old, now = time.time() - 400 * 86400, time.time()
    for n in range(150):
        ts = old + n
        store.save_cycle(ProcessCycle(f"m-{n}", "m", n, ts - 1, ts, curves={"p": np.arange(5.0)}), {"a": 1.0})
        store.save_inspection(InspectionRecord("", "c", ts, FAIL, cycle_id=f"m-{n}"))
    store.save_cycle(ProcessCycle("m-999", "m", 999, now - 1, now), {"a": 1.0})
    store.save_gap("m", old, 1, 2)
    before = store.cycle_features(limit=100, newest=False, machine_id="m")
    assert store.prune_curves(86400) == 50              # the baseline keeps its curves too
    done = store.prune(365 * 86400, batch=20)
    assert done["process_cycles"] == 50 and done["inspections"] == 50 and done["cycle_gaps"] == 1
    assert store.cycle_features(limit=100, newest=False, machine_id="m") == before
    assert store.get_cycle("m-999") is not None and store.get_cycle("m-0", with_curves=True).curves


def test_old_events_go_and_csv_files_turn_over(tmp_path, monkeypatch):
    import src.logging.event_logger as module
    from src.system.retention import RetentionWorker

    before, module.EventLogger._instance = module.EventLogger._instance, None
    try:
        log = module.EventLogger(str(tmp_path / "e.db"), str(tmp_path / "e.csv"))
        import sqlite3
        with sqlite3.connect(log.db_path) as conn:
            for age in (10, 100, 1000):
                stamp = time.time() - age * 86400
                conn.execute("INSERT INTO events (camera_id, timestamp) VALUES ('c', ?)", (stamp,))
                conn.execute("INSERT INTO defect_events (event_id, timestamp, source, defect_type, confidence, severity, "
                             "full_image_path) VALUES (?, ?, 'c', 'x', 1, 1, '')", (str(age), module.utc_naive(stamp).isoformat()))
        assert RetentionWorker({}, event_logger=log).run_once() == {"defect_events": 1, "events": 2, "geometry_events": 0}
        assert RetentionWorker({"logging": {"retention_days": 0, "tracking_retention_days": 0}},
                               event_logger=log).run_once()["defect_events"] == 0
        monkeypatch.setattr(module, "CSV_MAX_BYTES", 10)
        (tmp_path / "e.csv").write_text("x" * 50)
        log._rotate(str(tmp_path / "e.csv"))
        assert not (tmp_path / "e.csv").exists() and len(list(tmp_path.glob("e.*.csv"))) == 1
    finally:
        module.EventLogger._instance = before


@pytest.fixture
def audit_days(tmp_path, monkeypatch):
    import src.inference.audit_log as mod

    log = mod.InspectionAuditLog(tmp_path / "audit", tmp_path / "k")
    with monkeypatch.context() as m:
        for d in range(5):
            m.setattr(mod.time, "strftime", lambda fmt, *a, d=d: f"2026-01-{d + 1:02d}")
            for i in range(4):
                log.append({"part": i})
    return mod, tmp_path / "audit", tmp_path / "k", log


def test_archived_audit_days_verify_and_so_does_the_rest(audit_days, tmp_path):
    mod, directory, key, log = audit_days
    assert mod.archive(directory, "2026-01-03", tmp_path / "old", key) == {"moved": 2, "records": 8}
    assert mod.verify_chain(directory, key) == {"ok": True, "records": 12, "head": log._last_hash, "archived": 8}
    assert mod.verify_chain(tmp_path / "old", key)["records"] == 8
    log.append({"part": 9})                                          # the running writer carries on
    mod.InspectionAuditLog(directory, key).append({"part": 10})      # and so does a restarted one
    assert mod.verify_chain(directory, key)["records"] == 14
    assert mod.archive(directory, "2026-01-05", tmp_path / "old", key)["moved"] == 2    # added to the same archive
    assert mod.verify_chain(tmp_path / "old", key)["records"] == 16
    assert mod.archive(directory, "2099-01-01", tmp_path / "old", key)["moved"] in (0, 1)
    assert mod.verify_chain(directory, key)["ok"]


def test_archiving_cannot_hide_tampering(audit_days, tmp_path):
    mod, directory, key, _ = audit_days
    mod.archive(directory, "2026-01-03", tmp_path / "old", key)
    base = mod.base_path(directory)
    good = base.read_text()
    base.write_text(good.replace('"records": 8', '"records": 4'))
    assert not mod.verify_chain(directory, key)["ok"]
    base.unlink()
    assert not mod.verify_chain(directory, key)["ok"]
    base.write_text(good)
    old = tmp_path / "old" / "2026-01-01.jsonl"
    old.write_text(old.read_text().replace('"part": 3', '"part": 4'))
    assert not mod.verify_chain(tmp_path / "old", key)["ok"]
    day = directory / "2026-01-03.jsonl"
    day.write_text(day.read_text().replace('"part": 3', '"part": 4'))
    with pytest.raises(ValueError, match="not intact"):
        mod.archive(directory, "2026-01-04", tmp_path / "old2", key)


# -------------------------------------------------------------------------- Modbus, OPC UA, RTSP

class _Modbus:
    def __init__(self, refuse=(), dead=False):
        self.requests, self.refuse, self.dead = [], set(refuse), dead

    def read_holding_registers(self, address, count, device_id):
        self.requests.append((address, count))
        wanted = range(address, address + count)
        if self.dead:
            return type("E", (), {"isError": lambda s: True})()
        if self.refuse & set(wanted):
            return type("E", (), {"isError": lambda s: True, "exception_code": 2})()
        return type("R", (), {"isError": lambda s: False, "registers": [a % 1000 for a in wanted]})()


def _source():
    signals = [SignalSpec(f"s{i}", address=200 + 2 * i, data_type="uint16") for i in range(10)]
    return ModbusCycleSource("x", signals=signals, counter=SignalSpec("cycle_counter", address=100, data_type="uint16"),
                             setpoints=[SignalSpec("sp", address=400, data_type="uint16")])


def test_modbus_reads_blocks_not_signals():
    source, client = _source(), _Modbus()
    values = source.read_all(client)
    assert client.requests == [(100, 1), (200, 19), (400, 1)]
    assert values["s3"] == 206.0 and values["sp"] == 400.0 and values["cycle_counter"] == 100.0


def test_refused_register_costs_only_itself():
    source, client = _source(), _Modbus(refuse=[206])
    values = source.read_all(client)
    assert values["s3"] is None and values["s4"] == 208.0
    client.requests.clear()
    assert source.read_all(client)["s4"] == 208.0 and len(client.requests) == 12     # no failed block every sample
    client.refuse.clear()
    assert source.read_all(client)["s3"] == 206.0


def test_lost_modbus_connection_still_raises():
    with pytest.raises(IOError):
        _source().read_all(_Modbus(dead=True))


def test_long_and_mixed_ranges_respect_the_protocol_limit():
    specs = [SignalSpec(f"a{i}", address=10 * i, data_type="float32") for i in range(30)]
    specs += [SignalSpec("i", address=0, register_type="input")]
    for block in _source()._blocks(specs):
        assert max(s.address + 2 for s in block) - block[0].address <= 120
        assert len({s.register_type for s in block}) == 1


def test_signal_with_no_data_has_no_features_instead_of_zeros():
    assert curve_statistics(np.full(20, np.nan), 20.0) == {}
    full = np.sin(np.linspace(0, 3, 60)) + 5
    holed = full.copy()
    holed[10:14] = np.nan
    a, b = curve_statistics(full, 20.0), curve_statistics(holed, 20.0)
    assert b["integral"] == pytest.approx(a["integral"], rel=0.01)


def test_opcua_login_is_never_sent_unprotected():
    kw = dict(signals=[], counter=SignalSpec("c"), username="u", password="p")
    with pytest.raises(ValueError, match="unprotected"):
        OpcUaCycleSource("opc.tcp://x", **kw)
    with pytest.raises(ValueError, match="certificate"):
        OpcUaCycleSource("opc.tcp://x", security_policy="Basic256Sha256", **kw)
    secure = OpcUaCycleSource("opc.tcp://x", security_policy="Basic256Sha256", certificate="c.der", private_key="k.pem", **kw)
    assert secure.security == "Basic256Sha256,SignAndEncrypt,c.der,k.pem"
    OpcUaCycleSource("opc.tcp://x", allow_insecure_login=True, **kw)


def test_opcua_bad_quality_is_no_value():
    ua = pytest.importorskip("asyncua").ua
    source = OpcUaCycleSource("opc.tcp://x", signals=[], counter=SignalSpec("c"))
    spec = SignalSpec("p", scale=2.0)
    assert source._value(ua.DataValue(ua.Variant(3.0)), spec) == 6.0
    bad = ua.DataValue(ua.Variant(3.0), StatusCode=ua.StatusCode(ua.StatusCodes.BadSensorFailure))
    assert source._value(bad, spec) is None
    assert source._value(ua.DataValue(ua.Variant("text")), spec) is None
    assert source._value(ua.DataValue(ua.Variant(float("nan"))), spec) is None


def test_opcua_reads_a_real_server():
    asyncua = pytest.importorskip("asyncua")
    import asyncio

    async def go():
        server = asyncua.Server()
        await server.init()
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server.set_endpoint(f"opc.tcp://127.0.0.1:{port}")
        server.set_security_policy([asyncua.ua.SecurityPolicyType.NoSecurity])
        obj = await server.nodes.objects.add_object(2, "M")
        shot, pressure = await obj.add_variable(2, "Shot", 1), await obj.add_variable(2, "P", 5.0)
        async with server:
            source = OpcUaCycleSource(f"opc.tcp://127.0.0.1:{port}", poll_hz=50, machine_id="m",
                                      signals=[SignalSpec("p", node_id=pressure.nodeid.to_string())],
                                      counter=SignalSpec("cycle_counter", node_id=shot.nodeid.to_string()))
            cycles = []
            source.subscribe(cycles.append)
            task = asyncio.create_task(source._run_async())
            for n in range(1, 4):
                await shot.write_value(n)
                await asyncio.sleep(0.3)
                if n == 2:
                    bad = asyncua.ua.DataValue(asyncua.ua.Variant(9.0),
                                               StatusCode=asyncua.ua.StatusCode(asyncua.ua.StatusCodes.BadSensorFailure))
                    await pressure.write_value(bad)
            source._stop.set()
            await asyncio.wait_for(task, 5)
        return cycles
    cycles = asyncio.run(go())
    assert [c.sequence for c in cycles] == [1, 2] and np.nanmax(cycles[1].curves["p"]) == 5.0


def test_stream_that_never_answers_does_not_hold_the_thread():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(5)
        cm = CameraManager.__new__(CameraManager)
        cm.stream_timeout_s = 1.0
        started = time.monotonic()
        cap, _ = cm._open_camera(f"rtsp://127.0.0.1:{server.getsockname()[1]}/x")
        assert cap is None and time.monotonic() - started < 8
