"""Model operations: registration/tiling, ensembles, ONNX, registry, shadow mode, audit
trail, good-part pool, multi-view verdicts, labelling loop and the live inspection wiring."""
from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from src.inference.anomaly.base import AnomalyDetector
from src.process.synthetic_parts import BACKGROUND, render_part

DASHBOARD_APP = Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py"


def _jitter(img: np.ndarray, angle: float, dx: float, dy: float, *, interp=cv2.INTER_LINEAR) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 0.8)
    m[0, 2] += dx
    m[1, 2] += dy
    border = (BACKGROUND,) * 3 if img.ndim == 3 else 0
    return cv2.warpAffine(img, m, (w, h), flags=interp, borderValue=border)


class MeanImageDetector(AnomalyDetector):
    """Tiny deterministic detector: distance to the mean training view (no weights needed)."""

    kind = "mean-stub"

    def __init__(self, size: int = 96, device: str | None = "cpu"):
        super().__init__(device)
        self.size = size
        self.mean: np.ndarray | None = None
        self.trained_views: list[tuple[int, int]] = []
        self.scored = 0

    def _prep(self, img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return cv2.resize(g, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _train(self, images, **kwargs):
        self.trained_views = [im.shape[:2] for im in images]
        self.mean = np.mean([self._prep(im) for im in images], axis=0)
        return {"views": len(images)}

    def score_map(self, image):
        self.scored += 1
        d = np.abs(self._prep(image) - self.mean) / 255.0
        d = cv2.GaussianBlur(d, (0, 0), 1.5)
        full = cv2.resize(d, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        return float(d.max()), full.astype(np.float32)

    def _state(self):
        return {"size": self.size, "mean": self.mean.tolist()}

    @classmethod
    def _from_state(cls, state, device):
        det = cls(state["size"])
        det.mean = np.asarray(state["mean"], dtype=np.float32)
        return det


# ------------------------------------------------------------- registration

def test_registration_aligns_rotated_and_shifted_parts():
    from src.inference.anomaly.preprocess import Preprocessor, _template

    pp = Preprocessor(register=True)
    stats = pp.fit([render_part(320, 240, None, seed=s) for s in range(20)])
    assert stats["registered"] == 20
    ref = render_part(320, 240, None, seed=999)
    ref_t = _template(pp.views(ref)[0])
    rng = np.random.default_rng(0)
    for _ in range(8):
        moved = _jitter(ref, rng.uniform(-180, 180), *rng.uniform(-20, 20, 2))
        assert float((_template(pp.views(moved)[0]) * ref_t).mean()) > 0.97   # includes 180-degree flips


def test_registered_anomaly_map_points_at_the_defect_in_frame_coordinates():
    det = MeanImageDetector().configure_preprocessing(register=True)
    det.fit([render_part(320, 240, None, seed=s) for s in range(30)], holdout_fraction=0.3)
    img, mask = render_part(320, 240, "burn_mark", seed=7, return_mask=True)
    moved = _jitter(img, 35.0, 15, -10)
    moved_mask = _jitter(mask.astype(np.uint8) * 255, 35.0, 15, -10, interp=cv2.INTER_NEAREST) > 0
    result = det.predict(moved)
    assert result.is_anomalous and result.anomaly_map.shape == moved.shape[:2]
    y, x = np.unravel_index(np.argmax(result.anomaly_map), result.anomaly_map.shape)
    near = cv2.dilate(moved_mask.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    assert near[y, x]


def test_tiling_trains_on_tiles_and_stitches_a_full_map(tmp_path):
    from src.inference.anomaly.preprocess import tile_boxes

    boxes = tile_boxes(240, 320, 2, 3, 0.15)
    cover = np.zeros((240, 320), bool)
    for x1, y1, x2, y2 in boxes:
        cover[y1:y2, x1:x2] = True
    assert len(boxes) == 6 and cover.all()

    det = MeanImageDetector().configure_preprocessing(tiles=(2, 2))
    goods = [render_part(320, 240, None, seed=s) for s in range(20)]
    det.fit(goods, holdout_fraction=0.25)
    assert len(det.trained_views) == 4 * 15 and det.trained_views[0][0] < 240
    score, amap = det.infer(goods[0])
    assert amap.shape == (240, 320)
    det.save(tmp_path / "m.pt")
    payload = __import__("torch").load(tmp_path / "m.pt", weights_only=True)
    restored = MeanImageDetector._from_state(payload["state"], "cpu")
    restored._restore_common(payload)
    assert restored.preprocessor.tiles == (2, 2)
    assert restored.infer(goods[0])[0] == pytest.approx(score)


# ------------------------------------------------------------------ ensemble

def test_ensemble_normalises_members_on_a_disjoint_holdout_half():
    from src.inference.anomaly.ensemble import EnsembleDetector

    a, b = MeanImageDetector(64), MeanImageDetector(128)
    ens = EnsembleDetector([a, b])
    goods = [render_part(320, 240, None, seed=s) for s in range(50)]
    cal = ens.fit(goods, holdout_fraction=0.4)
    assert len(ens.norm) == 2 and all(scale > 0 for _, scale in ens.norm)
    assert cal["n_holdout"] == 10            # 20 held out: 10 normalise members, 10 calibrate the threshold
    bad = render_part(320, 240, "short_shot", seed=3)
    assert ens.predict(bad).normalized_score > ens.predict(render_part(320, 240, None, seed=777)).normalized_score


# ---------------------------------------------------------------------- ONNX

def test_patchcore_cnn_onnx_export_matches_pytorch(tmp_path):
    pytest.importorskip("onnxruntime")
    from src.inference.anomaly.export import export_onnx
    from src.inference.anomaly.registry import create_detector, load_detector

    det = create_detector("patchcore", backbone="resnet18", image_size=128, device="cpu")
    det.fit([render_part(160, 120, None, seed=s) for s in range(14)])
    check = [render_part(160, 120, d, seed=50 + i) for i, d in enumerate([None, "flash", "burn_mark"])]
    meta = export_onnx(det, tmp_path / "pc.onnx", check_images=check)
    assert meta["kind"] == "patchcore-onnx" and meta["score_max_rel_diff"] < 0.01
    served = load_detector(tmp_path / "pc.onnx")
    assert [served.predict(i).is_anomalous for i in check] == [det.predict(i).is_anomalous for i in check]


# ------------------------------------------------------------------ registry

def _fake_model(tmp_path, name="m.pt", content=b"weights"):
    p = tmp_path / name
    p.write_bytes(content)
    p.with_suffix(".card.json").write_text(json.dumps({"kind": "patchcore", "threshold": 1.0}))
    return p


def test_registry_versions_shadow_promote_and_rollback(tmp_path):
    from src.inference.model_registry import VisionModelRegistry

    reg = VisionModelRegistry(tmp_path / "reg")
    v1 = reg.register("housing", "anomaly", _fake_model(tmp_path, content=b"a"), by="alice")
    assert reg.resolve("housing", "anomaly")["active_version"] == v1      # first version becomes active
    v2 = reg.register("housing", "anomaly", _fake_model(tmp_path, content=b"b"), shadow=True, by="alice")
    r = reg.resolve("housing", "anomaly")
    assert (r["active_version"], r["shadow_version"]) == (v1, v2)
    assert r["shadow"].exists() and r["shadow"].with_suffix(".card.json").exists()
    assert reg.promote("housing", "anomaly", by="bob", reason="ok") == v2
    assert reg.resolve("housing", "anomaly")["shadow_version"] is None
    assert reg.rollback("housing", "anomaly", by="bob", reason="false rejects") == v1
    history = [h["action"] for h in reg.manifest("housing", "anomaly")["history"]]
    assert history == ["register", "activate", "register", "shadow", "activate", "rollback"]
    with pytest.raises(ValueError):
        reg.register("../evil", "anomaly", _fake_model(tmp_path))


def test_retrain_requests_are_deduplicated_and_need_a_decision(tmp_path):
    from src.inference.model_registry import VisionModelRegistry

    reg = VisionModelRegistry(tmp_path)
    req = reg.request_retrain("housing", camera_id="cam", reason="drift")
    assert req and reg.request_retrain("housing", camera_id="cam", reason="drift again") is None
    assert reg.decide(req["id"], approve=True, by="alice")["status"] == "approved"
    with pytest.raises(ValueError):
        reg.decide(req["id"], approve=False, by="bob")


def test_shadow_report_uses_human_review_labels(tmp_path):
    from src.inference.anomaly.monitoring import ReviewQueue
    from src.inference.model_registry import VisionModelRegistry

    reg = VisionModelRegistry(tmp_path / "reg")
    reg.register("housing", "anomaly", _fake_model(tmp_path, content=b"a"))
    v2 = reg.register("housing", "anomaly", _fake_model(tmp_path, content=b"b"), shadow=True)
    queue_ = ReviewQueue(tmp_path / "review")
    img = np.zeros((8, 8, 3), np.uint8)
    for i, (active, shadow, truth) in enumerate([("pass", "fail", "fail"), ("fail", "fail", "fail"),
                                                 ("fail", "pass", "pass"), ("pass", "pass", "pass")]):
        reg.log_shadow("housing", "anomaly", {"image_sha": f"sha{i}", "active_verdict": active,
                                               "shadow_verdict": shadow, "shadow_version": v2})
        meta_path = queue_.add("cam", img, normalized_score=1.0, verdict=active, image_sha=f"sha{i}")
        meta = json.loads(meta_path.read_text())
        meta["review"]["verdict"] = truth
        meta_path.write_text(json.dumps(meta))
    rep = reg.shadow_report("housing", review_dir=tmp_path / "review")
    assert rep["parts"] == 4 and rep["agreement"] == 0.5
    assert rep["reviewed"]["shadow_accuracy"] == 1.0 and rep["reviewed"]["active_accuracy"] == 0.5
    assert rep["reviewed"]["active_escapes"] == 1 and rep["reviewed"]["shadow_escapes"] == 0


# ------------------------------------------------- audit / good pool / multi-view

def test_audit_log_hash_chain_detects_tampering(tmp_path):
    from src.inference.audit_log import InspectionAuditLog, image_sha256, verify_chain

    log = InspectionAuditLog(tmp_path)
    for i in range(5):
        log.append({"camera_id": "cam", "verdict": "pass" if i % 2 else "fail",
                    "image_sha256": image_sha256(np.full((4, 4, 3), i, np.uint8))})
    assert verify_chain(tmp_path) == {"ok": True, "records": 5, "head": log._last_hash}
    InspectionAuditLog(tmp_path).append({"verdict": "pass"})   # a restarted writer continues the chain
    assert verify_chain(tmp_path)["records"] == 6
    f = next(tmp_path.glob("*.jsonl"))
    lines = f.read_text().splitlines()
    lines[1] = lines[1].replace('"verdict": "pass"', '"verdict": "fail"')
    f.write_text("\n".join(lines) + "\n")
    assert not verify_chain(tmp_path)["ok"]
    del lines[1]
    f.write_text("\n".join(lines) + "\n")
    assert "broken link" in verify_chain(tmp_path)["error"]


def test_good_pool_samples_confident_passes_and_evicts(tmp_path):
    from src.inference.good_pool import GoodPartPool

    pool = GoodPartPool(tmp_path, "housing", every_n=2, max_items=3, max_score=0.8)
    img = np.zeros((10, 10, 3), np.uint8)
    assert pool.offer(img, verdict="fail", normalized_score=0.1, camera_id="c") is None
    assert pool.offer(img, verdict="pass", normalized_score=0.95, camera_id="c") is None   # borderline: not pooled
    saved = [pool.offer(img, verdict="pass", normalized_score=0.3, camera_id="c") for _ in range(10)]
    assert sum(p is not None for p in saved) == 5 and len(pool) == 3
    assert len(pool.load(since_ts=time.time() + 60)) == 0 and len(pool.load()) == 3


def test_part_verdicts_fuse_views_and_flag_missing_ones(tmp_path):
    from src.process.models import InspectionRecord
    from src.process.store import ProcessStore

    store = ProcessStore(tmp_path / "p.db")
    rows = [("c1", "top", "pass", "vision"), ("c1", "bottom", "fail", "vision"),
            ("c2", "top", "pass", "vision"),
            ("c3", "top", "fail", "vision"), ("c3", "audit", "pass", "audit"),
            ("c4", "top", "pass", "vision"), ("c4", "bottom", "pass", "vision")]
    for i, (cycle, cam, verdict, source) in enumerate(rows):
        store.save_inspection(InspectionRecord(f"i{i}", cam, 100.0 + i, verdict, "flash" if verdict == "fail" else None,
                                               cycle_id=cycle, source=source))
    parts = {p["cycle_id"]: p for p in store.part_verdicts(0, 1e9, required_cameras=["top", "bottom"])}
    assert parts["c1"]["verdict"] == "fail" and parts["c1"]["defect_types"] == ["flash"]
    assert parts["c2"]["verdict"] == "incomplete" and parts["c2"]["missing_views"] == ["bottom"]
    assert parts["c3"]["verdict"] == "pass" and parts["c3"]["audited"]       # audit overrides vision
    assert parts["c4"]["verdict"] == "pass"
    store.close()


# ------------------------------------------------------------------ labelling

def test_labeling_export_prioritises_uncertain_parts_and_import_builds_dataset(tmp_path):
    from src.inference.anomaly.monitoring import ReviewQueue
    from src.inference.labeling import export_batch, import_labels

    q = ReviewQueue(tmp_path / "review")
    img = render_part(160, 120, None, seed=1)
    ids = {}
    for cam, score in (("a", 1.6), ("a", 1.02), ("b", 0.85), ("b", 1.2)):
        ids[score] = json.loads(q.add(cam, img, normalized_score=score, verdict="fail" if score > 1 else "pass",
                                      bbox=(10, 10, 40, 40), predicted_type="flash").read_text())["id"]
    out = export_batch(tmp_path / "review", tmp_path / "batch", fmt="labelstudio", class_names=["flash", "sink_mark"])
    assert out["items"] == 4
    batch = json.loads((tmp_path / "batch" / "batch.json").read_text())
    assert batch["review_ids"][:2] == [ids[1.02], ids[0.85]]   # closest to threshold first, cameras interleaved

    tasks = json.loads((tmp_path / "batch" / "tasks.json").read_text())
    for t in tasks:   # labeller: polygon on the 1.02 part, nothing elsewhere
        poly = [[10, 10], [40, 10], [40, 40]] if t["data"]["review_id"] == ids[1.02] else None
        t["annotations"] = [{"result": [] if poly is None else [{"type": "polygonlabels", "value": {
            "points": poly, "polygonlabels": ["sink_mark"]}}]}]
    (tmp_path / "export.json").write_text(json.dumps(tasks))
    res = import_labels(tmp_path / "export.json", tmp_path / "batch" / "images", tmp_path / "ds", fmt="labelstudio",
                        class_names=["flash", "sink_mark"], update_review=tmp_path / "review")
    assert res["defective_images"] == 1 and res["good_images"] == 3 and res["review_updated"] == 4
    labels = [p.read_text() for p in (tmp_path / "ds" / "labels").rglob("*.txt")]
    assert sum(bool(t) for t in labels) == 1 and any(t.startswith("1 0.1") for t in labels)
    reviewed = {json.loads(p.read_text())["id"]: json.loads(p.read_text())["review"]
                for p in (tmp_path / "review").rglob("*.json")}
    assert reviewed[ids[1.02]] == {"verdict": "fail", "defect_type": "sink_mark", "reviewer": "labeling-import"}
    assert reviewed[ids[1.6]]["verdict"] == "pass"


# ------------------------------------------------- live inspection wiring

def _health_fields(cm, views=()):
    """What the heartbeat reads; every view starts out delivering frames."""
    from src.camera.camera_manager import CameraWorkerState

    now = time.perf_counter()
    cm._lock = threading.Lock()
    cm._view_faults, cm._delivery_fault_until = {}, 0.0
    cm._heartbeat_write, cm._hold_reason = None, None
    cm.max_frame_age_s, cm.max_inspect_s = 5.0, 10.0
    cm.frame_detector_missing, cm._inspecting_since = False, None
    cm._inspection_slots = threading.Semaphore(4)
    cm.expect_part_per_cycle, cm._cycles_checked_to, cm._next_unseen_check = True, {}, 0.0
    cm.workers = {}
    for view in views or ("cam",):
        state = CameraWorkerState(view, "s", None, queue.Queue(), queue.Queue(), threading.Event())
        state.last_capture_ts = state.last_infer_ts = now
        cm.workers[view] = state


class _StubStack:
    def __init__(self, failed: bool, version: str, score: float):
        from src.inference.inspection_stack import PartJudgement

        self.empty = False
        self.versions = {"anomaly": {"version": version, "digest": version}}
        self.anomaly = type("A", (), {"threshold": 1.0, "calibration": {"holdout_mean": 0.5},
                                      "heatmap_overlay": staticmethod(lambda f, r, t: f)})()
        res = type("R", (), {"normalized_score": score, "is_anomalous": failed, "bbox": (1, 1, 5, 5), "score": score})()
        self._j = PartJudgement(failed, "anomaly" if failed else None, (1, 1, 5, 5) if failed else None,
                                0.7 if failed else None, res)

    def judge(self, frame, frame_defect_types=None):
        return self._j


def test_inspect_part_records_shadow_audit_review_and_pool(tmp_path):
    from src.camera.camera_manager import CameraManager, CameraWorkerState
    from src.camera.part_inspection import CompletedPart, PartVerdictFuser
    from src.inference.anomaly.monitoring import ReviewQueue
    from src.inference.audit_log import InspectionAuditLog, verify_chain
    from src.inference.model_registry import VisionModelRegistry

    reg = VisionModelRegistry(tmp_path / "reg")
    active, shadow = _StubStack(False, "v0001", 0.4), _StubStack(True, "v0002", 1.1)

    class Provider:
        registry = reg
        default_recipe = "housing"

        def get(self, recipe=None):
            return active, shadow

    cm = CameraManager.__new__(CameraManager)
    cm.stack_provider = Provider()
    cm.process_service = None
    cm.camera_recipes, cm.recipe_from_scalar = {}, None
    cm.audit_log = InspectionAuditLog(tmp_path / "audit")
    cm.review_queue = ReviewQueue(tmp_path / "review")
    cm._pool_cfg = {"dir": str(tmp_path / "pool"), "every_n": 1}
    cm._good_pools, cm.model_health, cm._health_cfg, cm.retrain_on_drift = {}, {}, {}, True
    recorded: list[Any] = []
    cm.fast_queue, cm.part_fuser = queue.Queue(), PartVerdictFuser()
    cm._part_counter, cm._inspection_faulted = 0, False
    _health_fields(cm)
    cm._record_new_defect = lambda *a, **k: recorded.append(k) or True
    cm._telemetry_context = lambda *a, **k: None

    state = CameraWorkerState("cam", "s", None, queue.Queue(), queue.Queue(), threading.Event())
    frame = render_part(160, 120, None, seed=3)
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, frame, 7, 3))
    assert state.parts_inspected == 1 and state.parts_failed == 0 and not recorded

    rep = reg.shadow_report("housing", review_dir=None, version="v0002")
    assert rep["parts"] == 1 and rep["shadow_only_rejects"] == 1
    audit = verify_chain(tmp_path / "audit")
    assert audit["ok"] and audit["records"] == 2    # the view's evidence + the part's final verdict
    lines = [json.loads(line) for line in next((tmp_path / "audit").glob("*.jsonl")).read_text().splitlines()]
    rec = next(r for r in lines if r.get("record") != "part_verdict")
    assert rec["models"]["anomaly"]["version"] == "v0001" and rec["shadow"]["verdict"] == "fail"
    assert rec["image_sha256"] and len(list((tmp_path / "pool" / "housing").glob("*.png"))) == 1
    assert cm._health_status["cam"]["models"] == {"anomaly": "v0001"}

    active._j.anomaly.normalized_score = 0.95       # borderline part -> review queue, not the good pool
    cm._inspect_part(state, CompletedPart("cam", 3.0, 4.0, frame, 8, 3))
    assert len(list((tmp_path / "review").rglob("*.json"))) == 1
    assert len(list((tmp_path / "pool" / "housing").glob("*.png"))) == 1


def test_stack_provider_hot_reloads_promoted_models(tmp_path):
    from src.inference.anomaly.registry import create_detector
    from src.inference.inspection_stack import StackProvider
    from src.inference.model_registry import VisionModelRegistry

    goods = [render_part(160, 120, None, seed=s) for s in range(12)]
    paths = []
    for i, size in enumerate((96, 128)):
        det = create_detector("patchcore", backbone="resnet18", image_size=size, device="cpu")
        det.fit(goods)
        det.save(tmp_path / f"m{i}.pt")
        paths.append(tmp_path / f"m{i}.pt")
    reg = VisionModelRegistry(tmp_path / "reg")
    reg.register("housing", "anomaly", paths[0])
    reg.register("housing", "anomaly", paths[1], shadow=True)
    provider = StackProvider({"registry_dir": str(tmp_path / "reg"), "recipe": "housing", "reload_check_s": 0})
    active, shadow = provider.get()
    assert active.versions["anomaly"]["version"] == "v0001" and shadow.versions["anomaly"]["version"] == "v0002"
    assert active.anomaly.image_size == 96 and shadow.anomaly.image_size == 128
    reg.promote("housing", "anomaly", by="alice", reason="test")
    active2, shadow2 = provider.get()
    assert active2.versions["anomaly"]["version"] == "v0002" and shadow2 is None
    reg.rollback("housing", "anomaly", by="alice", reason="test")
    assert provider.get()[0].versions["anomaly"]["version"] == "v0001"


def test_approved_retrain_trains_on_the_pool_and_registers_a_shadow(tmp_path):
    from src.inference.good_pool import GoodPartPool
    from src.inference.model_registry import VisionModelRegistry, main

    pool = GoodPartPool(tmp_path / "pool", "housing", every_n=1)
    for s in range(16):
        pool.offer(render_part(160, 120, None, seed=s), verdict="pass", normalized_score=0.2, camera_id="cam")
    reg = VisionModelRegistry(tmp_path / "reg")
    req = reg.request_retrain("housing", camera_id="cam", reason="drift", since_ts=0.0)
    with pytest.raises(ValueError):   # not approved yet
        main(["--root", str(tmp_path / "reg"), "retrain", req["id"], "--pool-dir", str(tmp_path / "pool"),
              "--min-images", "10", "--candidate", "patchcore_wrn50", "--work-dir", str(tmp_path / "work")])
    reg.decide(req["id"], approve=True, by="alice")
    result = main(["--root", str(tmp_path / "reg"), "retrain", req["id"], "--pool-dir", str(tmp_path / "pool"),
                   "--min-images", "10", "--candidate", "patchcore_wrn50", "--confidence", "0.5",
                   "--work-dir", str(tmp_path / "work")])
    assert result["images"] == 16
    assert reg.resolve("housing", "anomaly")["shadow_version"] == result["version"]
    assert reg.requests()[0]["status"] == "done"




def _reject_path_manager(tmp_path, stack):
    from src.camera.camera_manager import CameraManager, CameraWorkerState
    from src.camera.part_inspection import PartVerdictFuser
    from src.inference.audit_log import InspectionAuditLog

    class Provider:
        registry = None
        default_recipe = "housing"

        def get(self, recipe=None):
            return stack, None

    cm = CameraManager.__new__(CameraManager)
    cm.stack_provider = Provider()
    cm.process_service = None
    cm.camera_recipes, cm.recipe_from_scalar = {}, None
    cm.audit_log = InspectionAuditLog(tmp_path / "audit")
    cm.review_queue = None
    cm._pool_cfg = {"dir": str(tmp_path / "pool"), "every_n": 1}
    cm._good_pools, cm.model_health, cm._health_cfg, cm.retrain_on_drift = {}, {}, {}, False
    cm.fast_queue = queue.Queue(maxsize=20)
    cm.part_fuser = PartVerdictFuser()
    cm._part_counter, cm._inspection_faulted = 0, False
    _health_fields(cm)
    cm._record_new_defect = lambda *a, **k: True
    cm._telemetry_context = lambda *a, **k: None
    return cm, CameraWorkerState("cam", "s", None, queue.Queue(), queue.Queue(), threading.Event())


def test_failed_part_raises_the_advisory_plc_flag(tmp_path):
    from src.camera.part_inspection import CompletedPart

    cm, state = _reject_path_manager(tmp_path, _StubStack(True, "v0001", 1.4))
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert state.parts_failed == 1
    signal = cm.fast_queue.get_nowait()
    assert signal["verdict"] == "fail" and 1 <= signal["flag"] <= 3


def test_passed_part_reports_a_clear_flag_with_its_number(tmp_path):
    from src.camera.part_inspection import CompletedPart

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v0001", 0.4))
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    signal = cm.fast_queue.get_nowait()
    assert state.parts_inspected == 1 and signal["verdict"] == "pass" and signal["flag"] == 0 and signal["part"] == 1


def test_part_with_no_model_is_not_passed(tmp_path):
    from src.camera.part_inspection import CompletedPart
    from src.inference.inspection_stack import InspectionStack
    from src.telemetry.plc_adapter import PLC_FLAG_NOT_INSPECTED

    empty = InspectionStack("housing", anomaly_path=tmp_path / "missing.pt")
    assert empty.empty
    cm, state = _reject_path_manager(tmp_path, empty)
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))

    assert (state.parts_inspected, state.parts_failed, state.parts_not_inspected) == (0, 0, 1)
    assert not (tmp_path / "pool").exists() or not list((tmp_path / "pool").rglob("*.png"))
    signal = cm.fast_queue.get_nowait()
    assert signal["verdict"] == "not_inspected" and signal["flag"] == PLC_FLAG_NOT_INSPECTED
    records = [json.loads(line) for f in (tmp_path / "audit").glob("*.jsonl") for line in f.read_text().splitlines()]
    assert {r["verdict"] for r in records} == {"not_inspected"}


def test_image_quality_flags_blur_and_noise_but_not_good_or_defective_parts():
    from src.inference.anomaly.image_quality import image_fault, image_quality_baseline
    from src.process.simulator import DEFECT_TYPES

    good = [render_part(640, 480, None, seed=i) for i in range(10)]
    baseline = image_quality_baseline(good)
    assert all(image_fault(g, baseline) is None for g in good)
    assert all(image_fault(render_part(640, 480, d, seed=5), baseline) is None for d in DEFECT_TYPES)
    assert "sharpness" in image_fault(cv2.GaussianBlur(good[0], (0, 0), 1.5), baseline)
    noisy = good[0].astype(float) + np.random.default_rng(0).normal(0, 8, good[0].shape)
    assert "noise" in image_fault(np.clip(noisy, 0, 255).astype(np.uint8), baseline)
    assert image_fault(good[0], None) is None


def test_blurred_frame_is_a_camera_fault_not_a_reject(tmp_path):
    from src.camera.part_inspection import CompletedPart
    from src.inference.anomaly.image_quality import image_quality_baseline

    good = [render_part(160, 120, None, seed=i) for i in range(5)]
    stack = _StubStack(True, "v0001", 5.0)        # the model would call this part defective
    stack.anomaly.calibration = {"image_quality": image_quality_baseline(good)}
    cm, state = _reject_path_manager(tmp_path, stack)
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, cv2.GaussianBlur(good[0], (0, 0), 3.0), 7, 3))
    assert (state.parts_failed, state.parts_not_inspected) == (0, 1)
    assert cm.fast_queue.get_nowait()["verdict"] == "not_inspected" and cm._inspection_faulted


def test_dead_camera_stops_the_heartbeat_and_recovery_restarts_it(tmp_path):
    from src.camera.part_inspection import PartVerdictFuser

    class Adapter:
        beats = 0

        def write_heartbeat(self):
            Adapter.beats += 1

    from concurrent.futures import ThreadPoolExecutor

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v0001", 0.4))
    cm.part_fuser = PartVerdictFuser(["top", "side"], timeout_s=0.0)
    cm.plc_executor, cm.heartbeat_s, cm._last_heartbeat = ThreadPoolExecutor(1), 0.0, 0.0
    _health_fields(cm, ["top", "side"])

    state.camera_id = "top"
    cm._report_view(state, "c1", 0)                   # "side" never reports
    cm._reflex_tick(Adapter())
    assert cm._inspection_faulted and Adapter.beats == 0

    cm._report_view(state, "c2", 0)
    state.camera_id = "side"
    cm._report_view(state, "c2", 0)                   # both views report again
    cm._reflex_tick(Adapter())
    cm.plc_executor.shutdown(wait=True)
    assert not cm._inspection_faulted and Adapter.beats == 1


def test_inspection_crash_flags_the_part_instead_of_losing_it(tmp_path):
    from src.camera.part_inspection import CompletedPart

    class Broken(_StubStack):
        def judge(self, frame, frame_defect_types=None):
            raise RuntimeError("out of memory")

    cm, state = _reject_path_manager(tmp_path, Broken(False, "v0001", 0.4))
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert state.parts_not_inspected == 1
    assert cm.fast_queue.get_nowait()["verdict"] == "not_inspected"


def _modbus_recorder():
    """A real TCP server that answers Modbus writes and records them."""
    import socket
    import struct

    server = socket.create_server(("127.0.0.1", 0))
    frames: list[tuple[int, int, list[int]]] = []

    def serve():
        conn, _ = server.accept()
        with conn:
            while True:
                header = conn.recv(7)
                if len(header) < 7:
                    return
                body = conn.recv(struct.unpack(">H", header[4:6])[0] - 1)
                function, address = body[0], struct.unpack(">H", body[1:3])[0]
                if function == 0x06:
                    frames.append((function, address, [struct.unpack(">H", body[3:5])[0]]))
                    conn.sendall(header + body)
                else:  # 0x10 write multiple
                    count = struct.unpack(">H", body[3:5])[0]
                    frames.append((function, address, list(struct.unpack(f">{count}H", body[6:6 + 2 * count]))))
                    reply = body[:5]
                    conn.sendall(header[:4] + struct.pack(">H", len(reply) + 1) + header[6:7] + reply)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    return server, frames, worker


def test_part_verdict_writes_flag_and_part_number_in_one_frame_on_a_real_socket():
    from src.telemetry.plc_adapter import PLCAdapter, PLCAdapterConfig

    server, frames, worker = _modbus_recorder()
    adapter = PLCAdapter(PLCAdapterConfig(mode="plc", endpoint="127.0.0.1", port=server.getsockname()[1]))
    assert adapter.write_part_verdict({"verdict": "fail", "flag": 3, "part": 70001})
    assert adapter.write_heartbeat() and adapter.write_heartbeat()
    assert adapter.write_advisory_flag({"type": "GEOMETRIC_CRITICAL", "severity": 2})
    adapter.close()
    worker.join(timeout=2)
    server.close()
    assert frames == [(0x10, 100, [3, 70001 & 0xFFFF]), (0x06, 102, [1]), (0x06, 102, [2]), (0x06, 103, [2])]


def test_plc_write_failure_is_counted_not_raised():
    from src.telemetry.plc_adapter import PLCAdapter, PLCAdapterConfig

    adapter = PLCAdapter(PLCAdapterConfig(mode="plc", endpoint="127.0.0.1", port=1))
    assert adapter.write_heartbeat() is False and adapter.write_failures == 1


def test_fuser_gives_one_verdict_per_part_and_times_out_a_dead_camera():
    from src.camera.part_inspection import PartVerdictFuser

    fuser = PartVerdictFuser(["top", "side"], timeout_s=5.0)
    assert fuser.add("c1", "top", 0, 0.0) is None
    assert fuser.add("c1", "side", 0, 1.0)["verdict"] == "pass"

    assert fuser.add("c2", "top", 0, 2.0) is None
    failed = fuser.add("c2", "side", 2, 3.0)
    assert failed["verdict"] == "fail" and failed["flag"] == 2

    assert fuser.add("c3", "top", 0, 10.0) is None
    assert fuser.expired(14.9) == []
    late = fuser.expired(15.0)
    assert late[0]["verdict"] == "not_inspected" and late[0]["flag"] == 4 and late[0]["missing"] == ["side"]
    assert fuser.add("c3", "side", 0, 16.0) is None       # too late: no second verdict

    assert fuser.add("c4", "top", 1, 20.0) is None        # a timed-out view still cannot hide a fail
    assert fuser.expired(25.0)[0]["verdict"] == "fail"


def test_fuser_without_required_views_is_a_single_camera_station():
    from src.camera.part_inspection import PartVerdictFuser

    fuser = PartVerdictFuser()
    assert fuser.add(None, "cam", 0, 0.0)["verdict"] == "pass"
    assert fuser.add("c1", "cam", 3, 0.0)["flag"] == 3


def test_unlinked_view_of_a_multi_view_part_is_never_a_pass():
    from src.camera.part_inspection import PartVerdictFuser

    fuser = PartVerdictFuser(["top", "side"], timeout_s=5.0)
    assert fuser.add(None, "top", 0, 0.0) is None
    assert fuser.add(None, "side", 0, 0.1)["verdict"] == "not_inspected"   # both passed, still unproven
    assert fuser.add(None, "top", 0, 1.0) is None
    assert fuser.expired(6.1)[0]["verdict"] == "not_inspected"


def test_process_feed_down_keeps_one_verdict_per_part_and_rejects_still_reject():
    from src.camera.part_inspection import PartVerdictFuser

    fuser = PartVerdictFuser(["a", "b", "c"], timeout_s=5.0)
    out = [fuser.add(None, view, 2 if view == "b" else 0, 0.0) for view in "abc"]
    assert [o is not None for o in out] == [False, False, True]
    assert out[-1]["verdict"] == "fail" and out[-1]["cycle_id"] is None


def test_second_result_from_a_view_cannot_erase_a_reject():
    from src.camera.part_inspection import PartVerdictFuser

    fuser = PartVerdictFuser(["a", "b"], timeout_s=5.0)
    fuser.add("c1", "a", 3, 0.0)
    fuser.add("c1", "a", 0, 0.1)
    assert fuser.add("c1", "b", 0, 0.2)["flag"] == 3


def _audit_with(tmp_path, n=5):
    from src.inference.audit_log import InspectionAuditLog, verify_chain

    directory = tmp_path / "audit"
    log = InspectionAuditLog(directory)
    for i in range(n):
        log.append({"part": i, "verdict": "pass"})
    assert verify_chain(directory) == {"ok": True, "records": n, "head": log._last_hash}
    return directory, verify_chain


def test_audit_detects_deleting_the_newest_records(tmp_path):
    directory, verify = _audit_with(tmp_path)
    log_file = next(directory.glob("*.jsonl"))
    log_file.write_text("\n".join(log_file.read_text().splitlines()[:3]) + "\n")
    result = verify(directory)
    assert not result["ok"] and "2 newest records are missing" in result["error"]


def test_audit_detects_a_fully_rebuilt_chain_without_the_key(tmp_path):
    import hashlib

    directory, verify = _audit_with(tmp_path)
    log_file = next(directory.glob("*.jsonl"))
    prev, lines = "0" * 64, []
    for line in log_file.read_text().splitlines():
        rec = json.loads(line)
        rec["verdict"], rec["prev_hash"] = "fail", prev
        body = {k: v for k, v in rec.items() if k != "hash"}
        rec["hash"] = prev = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        lines.append(json.dumps(rec, sort_keys=True))
    log_file.write_text("\n".join(lines) + "\n")
    assert not verify(directory)["ok"]


def test_audit_detects_a_deleted_or_edited_anchor(tmp_path):
    from src.inference.audit_log import anchor_path

    directory, verify = _audit_with(tmp_path)
    anchor = anchor_path(directory)
    payload = json.loads(anchor.read_text())
    anchor.write_text(json.dumps({**payload, "records": 2}))
    assert "altered" in verify(directory)["error"]
    anchor.unlink()
    assert "anchor file missing" in verify(directory)["error"]


def test_audit_chain_survives_records_stamped_before_the_current_day(tmp_path):
    from src.inference.audit_log import InspectionAuditLog, verify_chain

    log = InspectionAuditLog(tmp_path / "audit")
    log.append({"ts": time.time(), "verdict": "pass"})
    log.append({"ts": time.time() - 3 * 86400, "verdict": "pass"})
    assert verify_chain(tmp_path / "audit")["records"] == 2


def test_audit_log_continues_its_chain_after_a_restart(tmp_path):
    from src.inference.audit_log import InspectionAuditLog

    directory, verify = _audit_with(tmp_path)
    InspectionAuditLog(directory).append({"part": 9, "verdict": "pass"})
    assert verify(directory)["records"] == 6


def test_accounts_check_password_lock_out_and_never_store_it(tmp_path):
    from src.operations.auth import LOCKOUT_S, MAX_FAILURES, UserStore

    path = tmp_path / "users.json"
    store = UserStore(path)
    assert not store.has_users()
    with pytest.raises(ValueError):
        store.add("ravi", "short", "QA")
    with pytest.raises(ValueError):
        store.add("ravi", "long enough password", "owner")
    store.add("ravi", "long enough password", "QA")
    assert "long enough password" not in path.read_text() and oct(path.stat().st_mode)[-3:] == "600"

    assert store.authenticate("ravi", "long enough password", now=0.0) == "QA"
    assert store.authenticate("nobody", "long enough password", now=0.0) is None
    for _ in range(MAX_FAILURES):
        assert store.authenticate("ravi", "wrong password!", now=10.0) is None
    assert store.authenticate("ravi", "long enough password", now=11.0) is None
    assert store.authenticate("ravi", "long enough password", now=10.0 + LOCKOUT_S + 1) == "QA"


def test_dashboard_needs_a_login_once_accounts_exist(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.operations.auth import UserStore

    users = tmp_path / "users.json"
    UserStore(users).add("ravi", "long enough password", "QA")
    config = tmp_path / "config.yaml"
    config.write_text(f"logging: {{sqlite_path: {tmp_path}/events.db}}\ndashboard: {{users_file: {users}}}\n")
    monkeypatch.setenv("CONFIG_PATH", str(config))

    at = AppTest.from_file(str(DASHBOARD_APP), default_timeout=30).run()
    assert not at.exception and at.subheader[0].value == "Sign in" and not at.metric
    at.text_input[0].set_value("ravi")
    at.text_input[1].set_value("wrong password!")
    at.button[0].click()
    at.run()
    assert at.error and "auth" not in at.session_state
    at.text_input[1].set_value("long enough password")
    at.button[0].click()
    at.run()
    assert at.session_state["auth"]["role"] == "QA" and not at.exception


def test_dashboard_renders_machine_health_and_position_from_real_records(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.logging.event_logger import EventLogger
    from src.process.features import CycleFeatureExtractor
    from src.process.models import InspectionRecord
    from src.process.simulator import InjectionMoldingSimulator
    from src.process.store import ProcessStore

    EventLogger(sqlite_path=str(tmp_path / "events.db"), csv_path=str(tmp_path / "e.csv"))._init_db()
    store = ProcessStore(tmp_path / "process.db")
    sim, extractor, now = InjectionMoldingSimulator(seed=3), CycleFeatureExtractor(), time.time()
    for i in range(60):
        cycle, _ = sim.next_cycle(start_ts=now - 600 + i)
        store.save_cycle(cycle, extractor.extract(cycle))
        store.save_inspection(InspectionRecord("", "cam", cycle.end_ts, "pass", cycle_id=cycle.cycle_id,
                                               pos_x=300.0 + i, pos_y=200.0, pos_angle=0.0))
    store.close()
    config = tmp_path / "config.yaml"
    config.write_text(f"logging: {{sqlite_path: {tmp_path}/events.db}}\n"
                      f"process: {{enabled: true, db_path: {tmp_path}/process.db}}\n"
                      f"dashboard: {{users_file: {tmp_path}/users.json}}\n"
                      f"esg_params: {{enabled: false}}\n")
    monkeypatch.setenv("CONFIG_PATH", str(config))

    at = AppTest.from_file(str(DASHBOARD_APP), default_timeout=60).run()
    assert not at.exception, [e.message for e in at.exception]
    assert any("/100" in str(m.value) for m in at.metric)
    assert "**Part-position drift**" in [m.value for m in at.markdown]


def test_bundled_vit_backbone_loads_with_no_hub_access(tmp_path, monkeypatch):
    from transformers import Dinov2Config, Dinov2Model

    from src.inference.anomaly.backbones import ViTBackbone

    tiny = Dinov2Config(hidden_size=24, num_hidden_layers=2, num_attention_heads=2, image_size=56, patch_size=14)
    Dinov2Model(tiny).save_pretrained(tmp_path / "dinov2_vits14")
    monkeypatch.setenv("QUALYXUS_BACKBONE_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty"))
    assert ViTBackbone("dinov2_vits14", device=torch.device("cpu")).embed_dim == 24


def test_machine_health_follows_hidden_wear_in_the_simulator():
    from src.process.features import CycleFeatureExtractor
    from src.process.machine_health import assess_machine
    from src.process.simulator import InjectionMoldingSimulator

    sim, extractor = InjectionMoldingSimulator(seed=11), CycleFeatureExtractor()
    rows, truth = [], {"Heater bands": [], "Check ring": [], "Mould cooling": []}
    for i in range(4000):
        cycle, outcome = sim.next_cycle(start_ts=float(i))
        rows.append(extractor.extract(cycle))
        truth["Heater bands"].append(-outcome.hidden_state["heater_offset"])
        truth["Check ring"].append(outcome.hidden_state["check_ring_wear"])
        truth["Mould cooling"].append(outcome.hidden_state["cooling_fouling"])
    health = {c.component: c for c in assess_machine(rows[:100], rows)}
    for name, wear in truth.items():
        assert np.corrcoef(health[name].history, wear)[0, 1] > 0.85, name
    assert {c.status for c in health.values()} <= {"OK", "WATCH", "ACT"}


def test_machine_health_ignores_an_intentional_setpoint_change_and_missing_signals():
    from src.process.machine_health import assess_component, assess_machine

    spec = {"signal": "melt", "setpoint": "sp", "direction": -1.0}
    rng = np.random.default_rng(0)
    base = [{"melt": 230 + rng.normal(0, 0.8), "sp": 230.0} for _ in range(100)]
    moved = [{"melt": 245 + rng.normal(0, 0.8), "sp": 245.0} for _ in range(100)]
    assert assess_component("x", spec, base, moved).status == "OK"
    worn = [{"melt": 212 + rng.normal(0, 0.8), "sp": 230.0} for _ in range(100)]
    assert assess_component("x", spec, base, worn).status == "ACT"
    assert assess_machine(base, moved, {"y": {"signal": "not_reported"}}) == []


def test_part_pose_reads_position_and_angle_from_the_frame():
    from src.camera.part_inspection import part_pose

    frame = np.full((480, 640, 3), 40, np.uint8)
    box = cv2.boxPoints(((400.0, 200.0), (300.0, 120.0), 10.0)).astype(np.int32)
    cv2.fillPoly(frame, [box], (180, 180, 180))
    x, y, angle = part_pose(frame)
    assert abs(x - 400) < 3 and abs(y - 200) < 3 and abs(angle - 10) < 2
    assert part_pose(np.full((480, 640, 3), 40, np.uint8)) is None


def test_part_position_drift_is_reported_per_camera(tmp_path, monkeypatch):
    from src.dashboard.data_access import fetch_part_position
    from src.process.models import InspectionRecord
    from src.process.store import ProcessStore

    store = ProcessStore(tmp_path / "p.db")
    for i in range(80):
        x = 300.0 if i < 50 else 300.0 + (i - 50)
        store.save_inspection(InspectionRecord("", "cam", float(i), "pass", pos_x=x, pos_y=200.0, pos_angle=0.0))
    store.close()
    summary = fetch_part_position(str(tmp_path / "p.db"))["summary"][0]
    assert summary["camera"] == "cam" and summary["status"] == "WATCH" and summary["drift_px"] > 15


def test_one_view_catches_a_defect_on_its_face_only():
    from src.process.synthetic_parts import SyntheticPartFeed, _ScheduledPart

    part = _ScheduledPart("c1", "short_shot", 0.0, 1.0, seed=8)   # 8 % 6 -> the view with index 2
    frames = {}
    for view in range(6):
        feed = SyntheticPartFeed(None, n_views=6, view_index=view)
        feed.part_in_view = lambda now: part
        frames[view] = feed.render(0.5, 160, 120)[0]
    clean = {}
    for view in range(6):
        feed = SyntheticPartFeed(None, n_views=6, view_index=view)
        feed.part_in_view = lambda now: _ScheduledPart("c1", None, 0.0, 1.0, seed=8)
        clean[view] = feed.render(0.5, 160, 120)[0]
    changed = [v for v in frames if np.abs(frames[v].astype(int) - clean[v].astype(int)).max() > 40]
    assert changed == [2]


def test_config_extends_overlays_a_base_file(tmp_path):
    from src.config.config import Config

    (tmp_path / "base.yaml").write_text("a: {x: 1, y: 2}\ncameras: [{id: old}]\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\na: {y: 3}\ncameras: [{id: new}]\n")
    raw = Config.load(config_path=str(tmp_path / "child.yaml")).raw
    assert raw["a"] == {"x": 1, "y": 3} and raw["cameras"] == [{"id": "new"}] and "extends" not in raw


def test_patchcore_training_does_not_hold_the_raw_bank_on_the_device():
    from src.inference.anomaly.patchcore import greedy_coreset

    features = torch.randn(500, 256)
    idx = greedy_coreset(features, 0.1, device=torch.device("cpu"))
    assert idx.shape[0] == 50 and idx.unique().shape[0] == 50


def test_recalibration_splits_the_false_reject_budget_across_views():
    from src.inference.anomaly.recalibrate import recalibrate

    class Detector:
        calibration: dict = {}

        def _holdout_scores(self, images):
            return list(images)

        def calibrate(self, scores, target_frr):
            self.target = target_frr
            self.calibration = {}
            return {"threshold": 1.0}

    det = Detector()
    frames = [np.zeros((48, 64, 3), np.uint8), np.full((48, 64, 3), 9, np.uint8)]
    result = recalibrate(det, frames, part_frr=0.012, n_views=6)
    assert det.target == pytest.approx(0.002)
    assert result["n_views"] == 6 and det.calibration["part_frr_target"] == 0.012


def test_demo_defects_rotate_and_keep_real_process_defects():
    from src.process.synthetic_parts import SyntheticPartFeed

    feed = SyntheticPartFeed(None, inject_defect_every=5)
    assert feed._defect_for(3, None) is None and feed._defect_for(3, "flash") == "flash"
    assert len({feed._defect_for(n, None) for n in (5, 10, 15, 20)}) == 4


# ------------------------------------------------------- what the PLC is told when things break
class _Beats:
    def __init__(self):
        self.n = 0

    def write_heartbeat(self):
        self.n += 1
        return True


def _heartbeat_manager(tmp_path, views=("top", "side")):
    from concurrent.futures import ThreadPoolExecutor
    from src.camera.part_inspection import PartVerdictFuser

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v0001", 0.4))
    cm.part_fuser = PartVerdictFuser(views, timeout_s=60.0)
    _health_fields(cm, views)
    cm.plc_executor, cm.heartbeat_s, cm._last_heartbeat = ThreadPoolExecutor(1), 0.0, 0.0
    return cm, state


def _beats_after_tick(cm) -> int:
    beats = _Beats()
    cm._reflex_tick(beats)
    cm.plc_executor.shutdown(wait=True)
    return beats.n


def test_healthy_station_keeps_its_heartbeat(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    assert _beats_after_tick(cm) == 1


@pytest.mark.parametrize("break_station", [
    lambda cm: cm.workers.clear(),
    lambda cm: cm.workers.pop("side"),
    lambda cm: setattr(cm.workers["top"], "failed", True),
    lambda cm: setattr(cm.workers["top"], "last_capture_ts", 0.0),
    lambda cm: setattr(cm.workers["top"], "last_capture_ts", time.perf_counter() - 60),
    lambda cm: setattr(cm.workers["side"], "last_infer_ts", time.perf_counter() - 60),
], ids=["all-gone", "one-gone", "worker-failed", "never-started", "camera-silent", "inference-hung"])
def test_heartbeat_stops_when_no_part_will_ever_time_out(tmp_path, break_station):
    """With nothing reporting, no part opens, so the view timeout alone would never fire."""
    cm, _ = _heartbeat_manager(tmp_path)
    break_station(cm)
    assert _beats_after_tick(cm) == 0


def test_heartbeat_stops_on_a_single_camera_station_too(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path, ["cam"])
    cm.part_fuser.required = frozenset()
    cm.workers["cam"].last_capture_ts = time.perf_counter() - 60
    assert _beats_after_tick(cm) == 0


def test_a_good_view_does_not_clear_another_views_fault(tmp_path):
    from src.camera import camera_manager as module
    from src.camera.part_inspection import CompletedPart

    cm, state = _heartbeat_manager(tmp_path)
    real = module.image_fault
    frame = render_part(160, 120, None, seed=3)
    try:
        state.camera_id, module.image_fault = "top", lambda f, b: "blurred"
        cm._inspect_part(state, CompletedPart("top", 1.0, 2.0, frame, 7, 3))
        state.camera_id, module.image_fault = "side", lambda f, b: None
        cm._inspect_part(state, CompletedPart("side", 1.0, 2.0, frame, 8, 3))
    finally:
        module.image_fault = real
    assert cm._view_faults == {"top": True, "side": False}
    assert _beats_after_tick(cm) == 0


def _full_disk(*args, **kwargs):
    raise OSError(28, "No space left on device")


def test_full_disk_cannot_cost_a_reject(tmp_path):
    from src.camera.part_inspection import CompletedPart

    cm, state = _reject_path_manager(tmp_path, _StubStack(True, "v0001", 1.4))
    cm.audit_log.append = _full_disk
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert cm.fast_queue.get_nowait()["verdict"] == "fail"


def test_timed_out_part_reaches_the_plc_with_a_full_disk(tmp_path):
    cm, state = _heartbeat_manager(tmp_path)
    cm.part_fuser.timeout_s = 0.0
    state.camera_id = "top"
    cm._report_view(state, "c1", 3)
    cm.audit_log.append = _full_disk
    cm._reflex_tick(_Beats())
    cm.plc_executor.shutdown(wait=True)
    assert cm.fast_queue.get_nowait()["flag"] == 3


def test_failure_after_the_verdict_leaves_the_verdict_alone(tmp_path):
    from src.camera.part_inspection import CompletedPart

    class Aligned:
        cycle_id, cavity = "m-1", 0

    class Link:
        parts_per_cycle = 1

    class Aligner:
        link_for = staticmethod(lambda camera: Link)
        align = staticmethod(lambda camera, ts: Aligned)

    class Service:
        aligner = Aligner

        @staticmethod
        def record_inspection(*args, **kwargs):
            raise RuntimeError("database is locked")

    cm, state = _reject_path_manager(tmp_path, _StubStack(True, "v0001", 1.4))
    cm.process_service = Service
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    sent = [cm.fast_queue.get_nowait() for _ in range(cm.fast_queue.qsize())]
    assert [s["verdict"] for s in sent] == ["fail"]


@pytest.mark.parametrize("adapter_result", [False, RuntimeError("lost")], ids=["returned-false", "raised"])
def test_verdict_that_never_arrived_stops_the_heartbeat(tmp_path, adapter_result):
    cm, _ = _heartbeat_manager(tmp_path)

    class Plc(_Beats):
        def write_part_verdict(self, signal):
            if isinstance(adapter_result, Exception):
                raise adapter_result
            return adapter_result

    cm._deliver_verdict(Plc(), {"part": 7})
    assert _beats_after_tick(cm) == 0


def test_verdict_dropped_from_a_full_queue_stops_the_heartbeat(tmp_path):
    from src.camera.part_inspection import CompletedPart

    cm, state = _reject_path_manager(tmp_path, _StubStack(True, "v0001", 1.4))
    cm.fast_queue = queue.Queue(maxsize=1)
    cm.fast_queue.put_nowait({})
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert cm._heartbeat_hold(time.monotonic()) == "a part verdict did not reach the PLC"


def test_heartbeats_do_not_queue_up_behind_a_hung_plc(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    release = threading.Event()

    class Hung(_Beats):
        def write_heartbeat(self):
            release.wait(5)

    for _ in range(50):
        cm._reflex_tick(Hung())
    backlog = cm.plc_executor._work_queue.qsize()
    release.set()
    cm.plc_executor.shutdown(wait=True)
    assert backlog == 0


def test_clean_stop_does_not_leave_a_pass_on_the_plc():
    from src.telemetry.plc_adapter import PLC_FLAG_NOT_INSPECTED, PLC_REGISTERS, PLCAdapter

    plc, sent = PLCAdapter(), {}
    plc._send = lambda address, values: sent.update({address: values[0]}) or True
    plc.clear()
    assert sent == {PLC_REGISTERS["verdict"]: PLC_FLAG_NOT_INSPECTED, PLC_REGISTERS["alert"]: 0}


@pytest.mark.parametrize("severity,written", [(-1, 0), (70000, 0xFFFF), (2, 2)])
def test_alert_value_always_fits_a_register(severity, written):
    from src.telemetry.plc_adapter import PLCAdapter

    plc, sent = PLCAdapter(), []
    plc._send = lambda address, values: sent.extend(values) or True
    plc.write_advisory_flag({"severity": severity})
    assert sent == [written]


# ----------------------------------------------------------------- audit trail under real failures
def _only_log_file(directory):
    return next(directory.glob("*.jsonl"))


def test_restart_cannot_bury_deleted_records(tmp_path):
    from src.inference.audit_log import InspectionAuditLog, verify_chain

    directory, _ = _audit_with(tmp_path)
    log_file = _only_log_file(directory)
    log_file.write_text("".join(log_file.read_text().splitlines(keepends=True)[:3]))
    InspectionAuditLog(directory).append({"part": 9, "verdict": "pass"})
    result = verify_chain(directory)
    assert not result["ok"] and "2 newest records are missing" in result["error"]


def test_restart_cannot_bury_a_deleted_anchor(tmp_path):
    from src.inference.audit_log import InspectionAuditLog, anchor_path, verify_chain

    directory, _ = _audit_with(tmp_path)
    anchor_path(directory).unlink()
    InspectionAuditLog(directory).append({"part": 9, "verdict": "pass"})
    assert not verify_chain(directory)["ok"]


def test_power_cut_mid_write_is_not_tampering_and_logging_continues(tmp_path):
    from src.inference.audit_log import InspectionAuditLog, verify_chain

    directory, _ = _audit_with(tmp_path)
    log_file = _only_log_file(directory)
    with open(log_file, "a") as fh:
        fh.write('{"hash": "abc", "part": 5, "tru')
    assert verify_chain(directory)["torn_tail"] is True
    InspectionAuditLog(directory).append({"part": 9, "verdict": "pass"})
    result = verify_chain(directory)
    assert result["ok"] and result["records"] == 6 and "torn_tail" not in result
    assert '"tru' in log_file.with_suffix(".torn").read_text()


def test_corrupt_line_in_the_middle_is_reported_not_a_crash(tmp_path):
    from src.inference.audit_log import verify_chain

    directory, _ = _audit_with(tmp_path)
    log_file = _only_log_file(directory)
    lines = log_file.read_text().splitlines(keepends=True)
    lines[2] = "not json\n"
    log_file.write_text("".join(lines))
    assert verify_chain(directory)["error"].endswith(":3 is not a valid record")


def test_any_recordable_value_survives_verification(tmp_path):
    from src.inference.audit_log import InspectionAuditLog, verify_chain

    log = InspectionAuditLog(tmp_path / "audit")
    log.append({"counts": {10: "a", 9: "b"}, "score": np.float32(0.5), "n": np.int64(3), "box": (1, 2)})
    assert verify_chain(tmp_path / "audit")["ok"]


# ------------------------------------------------------------------------------ exposure limits
def test_streams_need_the_token_when_one_is_set(monkeypatch):
    from fastapi import HTTPException
    from starlette.requests import Request

    from src.api.pulse_server import require_token

    def request(header=None):
        headers = [(b"authorization", header.encode())] if header else []
        return Request({"type": "http", "headers": headers})

    require_token(request())                                  # no token configured
    monkeypatch.setenv("QUALYXUS_API_TOKEN", "s3cret")
    require_token(request("Bearer s3cret"))
    for wrong in (None, "Bearer nope", "s3cret"):
        with pytest.raises(HTTPException):
            require_token(request(wrong))


def test_api_is_closed_by_default():
    from src.api.pulse_server import PulseServer, app

    assert PulseServer().host == "127.0.0.1"
    cors = next(m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware")
    assert cors.kwargs["allow_origins"] == [] and not cors.kwargs.get("allow_credentials")


def test_report_template_escapes_recorded_text(tmp_path):
    from jinja2 import DictLoader, Environment, select_autoescape

    source = Path("src/compliance/report_generator.py").read_text(encoding="utf-8")
    assert 'autoescape=select_autoescape(["html"])' in source
    env = Environment(loader=DictLoader({"r.html": "{{ v }}"}), autoescape=select_autoescape(["html"]))
    assert "<script>" not in env.get_template("r.html").render(v="<script>x</script>")


def test_wall_clock_jumping_back_does_not_stall_the_heartbeat(tmp_path, monkeypatch):
    cm, _ = _heartbeat_manager(tmp_path)
    beats = _Beats()
    cm._reflex_tick(beats)
    cm._heartbeat_write.result()
    monkeypatch.setattr(time, "time", lambda: 0.0)       # clock corrected far backwards
    cm._reflex_tick(beats)
    cm.plc_executor.shutdown(wait=True)
    assert beats.n == 2


def test_corrupt_audit_key_is_reported_not_a_crash(tmp_path):
    from src.inference.audit_log import verify_chain

    directory, _ = _audit_with(tmp_path)
    directory.with_name(directory.name + ".key").write_text("not hex")
    result = verify_chain(directory)
    assert not result["ok"] and "unreadable" in result["error"]


def test_model_that_failed_to_load_makes_parts_not_inspected(tmp_path):
    from src.inference.inspection_stack import InspectionStack

    bad = tmp_path / "broken.pt"
    bad.write_bytes(b"not a model")
    stack = InspectionStack("r", anomaly_path=bad)
    stack.segmenter = object()                # the other model is fine: still only half a check
    assert stack.empty


# ---------------------------------------------------- parts that slip through the normal path
class _Machine:
    """Process side of a station: which cycles finished, and how they link to the camera."""

    def __init__(self, cycles, mode="time", parts_per_cycle=1):
        link = type("Link", (), {"machine_id": "m", "mode": mode, "transport_delay_s": 4.0, "tolerance_s": 2.0,
                                 "parts_per_cycle": parts_per_cycle})
        self.aligner = type("Aligner", (), {"links": {"cam": link}})
        self.store = type("Store", (), {"cycles_between": staticmethod(
            lambda machine, since, until: [(c, t) for c, t in cycles if since < t <= until])})


def _unseen_after(cm, service, wall_now=100.0):
    cm.process_service = service
    cm.part_fuser.timeout_s = 5.0
    cm._cycles_checked_to = {"m": 0.0}
    return cm._unseen_parts(wall_now)         # cycles up to t=87 are overdue


def test_cycle_no_camera_saw_is_reported_not_forgotten(tmp_path):
    cm, state = _heartbeat_manager(tmp_path)
    cm.plc_executor.shutdown()
    state.camera_id = "top"
    for view in ("top", "side"):
        state.camera_id = view
        cm._report_view(state, "m-1", 0)
    found = _unseen_after(cm, _Machine([("m-1", 50.0), ("m-2", 60.0), ("m-3", 95.0)]))
    assert [(v["cycle_id"], v["flag"], v["verdict"]) for v in found] == [("m-2", 4, "not_inspected")]
    assert found[0]["missing"] == ["side", "top"]


def test_unseen_part_is_reported_once_and_stops_the_heartbeat(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    service = _Machine([("m-2", 60.0)])
    for verdict in _unseen_after(cm, service):
        cm._publish_part_verdict(verdict)
    assert cm.fast_queue.qsize() == 1 and cm.part_fuser.unseen(["m-2"]) == []
    assert _beats_after_tick(cm) == 0


def test_each_missing_cavity_of_a_shot_gets_a_verdict(tmp_path):
    cm, state = _heartbeat_manager(tmp_path, ["cam"])
    cm.plc_executor.shutdown()
    cm.part_fuser.required = frozenset()
    state.camera_id = "cam"
    cm._report_view(state, "m-1#0", 0)
    found = _unseen_after(cm, _Machine([("m-1", 50.0)], parts_per_cycle=2))
    assert [v["cycle_id"] for v in found] == ["m-1#1"]


def test_unseen_check_is_off_where_timing_says_nothing(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    cm.plc_executor.shutdown()
    assert _unseen_after(cm, _Machine([("m-2", 60.0)], mode="fifo")) == []
    cm.expect_part_per_cycle, cm._next_unseen_check = False, 0.0
    assert _unseen_after(cm, _Machine([("m-2", 60.0)])) == []


def test_slow_box_reports_overflow_parts_instead_of_queueing_forever(tmp_path):
    from src.camera.part_inspection import PartVerdictFuser

    cm, state = _heartbeat_manager(tmp_path, ["cam"])
    cm.plc_executor.shutdown()
    cm.part_fuser = PartVerdictFuser()
    cm.inspection_enabled = True
    cm.presence_std_threshold, cm._presence_on_frames, cm._presence_off_frames = 3.0, 1, 1
    cm.inspection_executor = type("Busy", (), {"submit": staticmethod(lambda *a: None)})
    cm._inspection_slots = threading.Semaphore(0)
    part, empty = render_part(160, 120, None, seed=3), np.full((120, 160, 3), 40, np.uint8)
    cm.part_trackers = {}
    cm._track_part(state, part, 1, [], [])
    cm._track_part(state, empty, 2, [], [])
    assert cm.fast_queue.get_nowait()["verdict"] == "not_inspected"
    assert cm._view_faults["cam"] is True


def test_stuck_inspection_stops_the_heartbeat(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    cm._inspecting_since = time.monotonic() - 60
    assert _beats_after_tick(cm) == 0


def test_inspection_slot_is_returned_even_when_inspection_crashes(tmp_path):
    cm, state = _heartbeat_manager(tmp_path)
    cm.plc_executor.shutdown()
    slots = threading.Semaphore(1)
    slots.acquire()
    cm._inspection_slots = slots
    cm._inspect_part = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        cm._inspect_queued(state, None)
    assert slots.acquire(blocking=False) and cm._inspecting_since is None


def test_frame_detector_that_failed_to_load_makes_parts_not_inspected(tmp_path):
    from src.camera.part_inspection import CompletedPart

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v0001", 0.4))
    cm.frame_detector_missing = True
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert cm.fast_queue.get_nowait()["verdict"] == "not_inspected"


def test_yolo_stays_out_of_the_default_runtime(monkeypatch):
    import builtins

    from src.config.config import Config
    from src.inference.multiplexer import InferenceMultiplexer

    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] != "ultralytics", "default runtime imported ultralytics"
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    mux = InferenceMultiplexer(Config.load())
    mux.stop()
    assert mux.mock_mode and not mux.load_failed


def test_configured_frame_detector_that_is_missing_is_flagged(tmp_path):
    from src.inference.multiplexer import InferenceMultiplexer

    class Cfg:
        raw = {}

        @staticmethod
        def get(*keys, default=None):
            return {"model_path": str(tmp_path / "gone.pt"), "defect_classes": ["scratch"]}.get(keys[-1], default)

    mux = InferenceMultiplexer(Cfg())
    mux.stop()
    assert mux.load_failed


def test_busy_api_port_does_not_stop_inspection(monkeypatch):
    import asyncio

    import uvicorn

    from src.api.pulse_server import PulseServer

    async def cannot_bind(self):
        raise SystemExit(1)

    monkeypatch.setattr(uvicorn.Server, "serve", cannot_bind)
    asyncio.run(PulseServer().run_async())       # returns instead of ending the process


def test_vibration_overload_cannot_touch_waiting_verdicts(tmp_path):
    from src.camera.camera_manager import CameraManager
    from src.inference.analysis.vibration_engine import VibrationAnalyzer

    cm, _ = _heartbeat_manager(tmp_path)
    cm.plc_executor.shutdown()
    cm.alert_queue = queue.Queue(maxsize=1)
    analyzer = VibrationAnalyzer.__new__(VibrationAnalyzer)
    analyzer._fast_queue = cm.alert_queue
    cm.fast_queue.put_nowait({"type": "PART_VERDICT", "flag": 3})
    for _ in range(5):                                     # alert storm against a full queue
        analyzer._emit_vibration_critical_fast(dominant_freq=1.0, psd_peak=1.0, rms=1.0, peaks=[])
    assert cm.alert_queue.qsize() == 1 and cm.fast_queue.qsize() == 1
    assert CameraManager._next_reflex_signal(cm)["type"] == "PART_VERDICT"
    assert CameraManager._next_reflex_signal(cm)["type"] == "VIBRATION_CRITICAL"
    assert CameraManager._next_reflex_signal(cm) is None


def test_no_runtime_path_can_command_the_machine():
    """Advisory only: nothing here may publish a speed or setpoint command."""
    offenders = [str(p) for p in Path("src").rglob("*.py") if "THROTTLE_SPEED" in p.read_text(encoding="utf-8")]
    assert offenders == []


def _exploding_pickle(path):
    """A file that runs code when unpickled, as a swapped model could."""
    import os
    import pickle

    class Payload:
        def __reduce__(self):
            return os.mkdir, (str(path) + ".was_executed",)

    path.write_bytes(pickle.dumps(Payload()))


def test_swapped_model_file_is_refused_before_it_can_run(tmp_path):
    from src.inference import signed_model

    path = tmp_path / "risk.joblib"
    signed_model.save({"ok": 1}, path)
    assert signed_model.load(path) == {"ok": 1}
    _exploding_pickle(path)
    with pytest.raises(ValueError, match="does not match"):
        signed_model.load(path)
    assert not Path(f"{path}.was_executed").exists()


def test_unsigned_model_is_refused_until_someone_signs_it(tmp_path):
    from src.inference import signed_model

    path = tmp_path / "risk.joblib"
    _exploding_pickle(path)
    with pytest.raises(ValueError, match="no signature"):
        signed_model.load(path)
    assert not Path(f"{path}.was_executed").exists()


def test_signed_typing_model_survives_registration(tmp_path):
    from src.inference import signed_model
    from src.inference.model_registry import VisionModelRegistry

    model = tmp_path / "typer.joblib"
    signed_model.save({"ok": 1}, model)
    registry = VisionModelRegistry(tmp_path / "reg")
    version = registry.register("r", "typing", model)
    version = version if isinstance(version, str) else version["version"]
    assert signed_model.load(registry.path_of("r", "typing", version)) == {"ok": 1}
