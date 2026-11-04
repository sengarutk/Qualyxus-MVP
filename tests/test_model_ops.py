"""Model operations: registration/tiling, ensembles, ONNX, registry, shadow mode, audit
trail, good-part pool, multi-view verdicts, labelling loop and the live inspection wiring."""
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np
import pytest

from src.inference.anomaly.base import AnomalyDetector
from src.process.synthetic_parts import BACKGROUND, render_part


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
    from src.camera.part_inspection import CompletedPart
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
    cm._record_new_defect = lambda *a, **k: recorded.append(k) or True
    cm._telemetry_context = lambda *a, **k: None

    state = CameraWorkerState("cam", "s", None, queue.Queue(), queue.Queue(), threading.Event())
    frame = render_part(160, 120, None, seed=3)
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, frame, 7, 3))
    assert state.parts_inspected == 1 and state.parts_failed == 0 and not recorded

    rep = reg.shadow_report("housing", review_dir=None, version="v0002")
    assert rep["parts"] == 1 and rep["shadow_only_rejects"] == 1
    audit = verify_chain(tmp_path / "audit")
    assert audit["ok"] and audit["records"] == 1
    rec = json.loads(next((tmp_path / "audit").glob("*.jsonl")).read_text())
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
    cm._record_new_defect = lambda *a, **k: True
    cm._telemetry_context = lambda *a, **k: None
    return cm, CameraWorkerState("cam", "s", None, queue.Queue(), queue.Queue(), threading.Event())


def test_failed_part_raises_the_advisory_plc_flag(tmp_path):
    from src.camera.part_inspection import CompletedPart

    cm, state = _reject_path_manager(tmp_path, _StubStack(True, "v0001", 1.4))
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert state.parts_failed == 1
    signal = cm.fast_queue.get_nowait()
    assert signal["type"] == "PART_REJECT" and 1 <= signal["severity"] <= 3


def test_passed_part_raises_no_flag(tmp_path):
    from src.camera.part_inspection import CompletedPart

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v0001", 0.4))
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert state.parts_inspected == 1 and cm.fast_queue.empty()


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
    assert signal["type"] == "PART_NOT_INSPECTED" and signal["severity"] == PLC_FLAG_NOT_INSPECTED
    records = [json.loads(line) for f in (tmp_path / "audit").glob("*.jsonl") for line in f.read_text().splitlines()]
    assert [r["verdict"] for r in records] == ["not_inspected"]


def test_inspection_crash_flags_the_part_instead_of_losing_it(tmp_path):
    from src.camera.part_inspection import CompletedPart

    class Broken(_StubStack):
        def judge(self, frame, frame_defect_types=None):
            raise RuntimeError("out of memory")

    cm, state = _reject_path_manager(tmp_path, Broken(False, "v0001", 0.4))
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    assert state.parts_not_inspected == 1
    assert cm.fast_queue.get_nowait()["type"] == "PART_NOT_INSPECTED"


def test_advisory_flag_reaches_a_real_modbus_socket():
    """The reject flag crosses a real TCP socket as a Modbus 0x06 write, on the configured port."""
    import socket
    import struct

    from src.telemetry.plc_adapter import PLCAdapter, PLCAdapterConfig

    server = socket.create_server(("127.0.0.1", 0))
    port = server.getsockname()[1]
    writes: list[tuple[int, int, int]] = []

    def serve():
        conn, _ = server.accept()
        with conn:
            request = conn.recv(12)
            function, register, value = struct.unpack(">BHH", request[7:12])
            writes.append((function, register, value))
            conn.sendall(request)  # a Modbus 0x06 response echoes the request

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    adapter = PLCAdapter(PLCAdapterConfig(mode="plc", endpoint="127.0.0.1", port=port))
    assert adapter.write_advisory_flag({"type": "PART_REJECT", "severity": 3}) is True
    worker.join(timeout=2)
    server.close()
    assert writes == [(0x06, 100, 3)]


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


def test_recalibration_splits_the_false_reject_budget_across_views():
    from src.inference.anomaly.recalibrate import recalibrate

    class Detector:
        calibration: dict = {}

        def _holdout_scores(self, images):
            return list(images)

        def calibrate(self, scores, target_frr):
            self.target = target_frr
            self.calibration = {}
            return {"threshold": max(scores)}

    det = Detector()
    result = recalibrate(det, [0.1, 0.2], part_frr=0.012, n_views=6)
    assert det.target == pytest.approx(0.002)
    assert result["n_views"] == 6 and det.calibration["part_frr_target"] == 0.012


def test_demo_defects_rotate_and_keep_real_process_defects():
    from src.process.synthetic_parts import SyntheticPartFeed

    feed = SyntheticPartFeed(None, inject_defect_every=5)
    assert feed._defect_for(3, None) is None and feed._defect_for(3, "flash") == "flash"
    assert len({feed._defect_for(n, None) for n in (5, 10, 15, 20)}) == 4
