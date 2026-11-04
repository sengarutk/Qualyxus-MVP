from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from src.inference.anomaly.calibration import conformal_threshold
from src.inference.anomaly.metrics import aupro, frr_at_recall, image_metrics, pixel_metrics
from src.process.synthetic_parts import render_part


# ------------------------------------------------------------------ calibration

def test_conformal_threshold_bounds_false_rejects():
    rng = np.random.default_rng(0)
    alpha, trials, frrs = 0.05, 300, []
    for _ in range(trials):
        cal = rng.normal(size=199)
        t = conformal_threshold(cal, target_frr=alpha)
        assert t["frr_guaranteed"]
        frrs.append(np.mean(rng.normal(size=2000) > t["threshold"]))
    assert np.mean(frrs) <= alpha + 0.005   # marginal guarantee holds on average


def test_conformal_threshold_reports_when_no_guarantee_is_possible():
    t = conformal_threshold([0.1, 0.2, 0.3], target_frr=0.01)
    assert not t["frr_guaranteed"] and t["threshold"] > 0.3 and t["min_holdout_for_guarantee"] == 99


def test_conformal_confidence_bounds_this_deployments_false_rejects():
    rng = np.random.default_rng(1)
    alpha, conf, n, misses = 0.05, 0.9, 300, 0
    for _ in range(300):
        t = conformal_threshold(rng.normal(size=n), target_frr=alpha, confidence=conf)
        assert t["frr_guaranteed"] and t["frr_upper_bound_95"] is not None
        misses += (1.0 - _norm_cdf(t["threshold"])) > alpha     # exact realised FRR of this draw
    assert misses / 300 <= 1.0 - conf + 0.04                    # P(FRR > alpha) <= 1 - confidence
    small = conformal_threshold(rng.normal(size=100), target_frr=0.01, confidence=0.9)
    assert not small["frr_guaranteed"] and small["min_holdout_for_guarantee"] == 230


def _norm_cdf(x: float) -> float:
    import math

    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ---------------------------------------------------------------------- metrics

def test_image_metrics_and_frr_at_recall():
    labels = np.array([0, 0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.3, 0.5, 0.9])
    # Catching both defects needs threshold 0.5 -> good 0.6 is rejected: FRR 1/4.
    assert frr_at_recall(labels, scores, 1.0) == pytest.approx(0.25)
    m = image_metrics(labels, scores)
    assert m["image_auroc"] == pytest.approx(7 / 8)
    assert m["frr_at_recall_100"] == pytest.approx(0.25)


def test_aupro_is_one_for_perfect_maps_and_low_for_noise():
    rng = np.random.default_rng(0)
    gts = np.zeros((4, 64, 64), bool)
    gts[0, 10:20, 10:20] = True
    gts[1, 40:44, 40:60] = True
    perfect = gts.astype(np.float32)
    assert aupro(perfect, gts) == pytest.approx(1.0, abs=1e-6)
    noise = rng.random(gts.shape).astype(np.float32)
    assert aupro(noise, gts) < 0.3
    px = pixel_metrics(list(perfect), list(gts.astype(np.uint8)))
    assert px["pixel_auroc"] == pytest.approx(1.0)


# ------------------------------------------------------- datasets / pseudo-defects

def test_pseudo_defects_change_only_the_masked_region():
    from src.inference.anomaly.pseudo_defects import cutpaste, texture_blend

    rng = np.random.default_rng(1)
    img = render_part(320, 240, None, seed=1)
    for fn in (cutpaste, texture_blend):
        out, mask = fn(img, rng)
        assert mask.sum() > 0
        diff = np.abs(out.astype(int) - img.astype(int)).sum(axis=2) > 0
        grown = cv2.dilate(mask, np.ones((7, 7), np.uint8)) > 0
        assert not np.any(diff & ~grown)


def test_folder_dataset_loader(tmp_path):
    from src.inference.anomaly.datasets import load_folder

    for sub in ("train/good", "test/good", "test/flash", "ground_truth/flash"):
        (tmp_path / sub).mkdir(parents=True)
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"train/good/{i}.png"), render_part(160, 120, None, i))
        cv2.imwrite(str(tmp_path / f"test/good/{i}.png"), render_part(160, 120, None, 10 + i))
    img, mask = render_part(160, 120, "flash", 5, return_mask=True)
    cv2.imwrite(str(tmp_path / "test/flash/0.png"), img)
    cv2.imwrite(str(tmp_path / "ground_truth/flash/0_mask.png"), mask * 255)
    ds = load_folder(tmp_path)
    assert len(ds.train_good) == 3 and sum(ds.test_labels) == 1 and ds.has_masks
    assert ds.test_masks[ds.test_labels.index(1)].sum() == mask.sum()


def test_good_only_dataset_is_marked_proxy(tmp_path):
    from src.inference.anomaly.datasets import load_good_only

    for i in range(20):
        cv2.imwrite(str(tmp_path / f"{i}.png"), render_part(160, 120, None, i))
    ds = load_good_only(tmp_path)
    assert ds.proxy and sum(ds.test_labels) > 0 and ds.has_masks


# ------------------------------------------------------------ detectors / registry

def _syn(n_good=30):
    good = [render_part(320, 240, None, seed=1000 + i) for i in range(n_good)]
    test = [(render_part(320, 240, d, seed=7000 + i), int(d is not None))
            for i, d in enumerate([None] * 6 + ["short_shot", "flash", "sink_mark", "burn_mark"] * 2)]
    return good, test


@pytest.mark.parametrize("method,kwargs,train", [
    ("patchcore", {"backbone": "resnet18", "image_size": 160}, {}),
    ("patchcore", {"backbone": "dinov2_vits14", "image_size": 224}, {}),
    ("dinomaly", {"backbone": "dinov2reg_vits14", "image_size": 224}, {"iterations": 150, "log_every": 0}),
])
def test_detectors_learn_from_good_parts_and_roundtrip(tmp_path, method, kwargs, train):
    from src.inference.anomaly.registry import create_detector, load_detector

    good, test = _syn()
    det = create_detector(method, **kwargs)
    det.fit(good, **train)
    scores = [det.predict(img).score for img, _ in test]
    assert image_metrics([y for _, y in test], scores)["image_auroc"] > 0.9
    det.save(tmp_path / "m.pt")
    loaded = load_detector(tmp_path / "m.pt")
    assert loaded.kind == method
    assert loaded.predict(test[-1][0]).score == pytest.approx(scores[-1], rel=2e-2)


def test_legacy_patchcore_files_still_load(tmp_path):
    import torch

    from src.inference.anomaly.registry import load_detector

    torch.save({"format": "qualyxus-patchcore-v1", "backbone": "resnet18", "image_size": 160, "embed_dim": 64,
                "coreset_fraction": 0.1, "memory_bank": torch.randn(50, 64).half(), "threshold": 1.0,
                "calibration": {}}, tmp_path / "legacy.pt")
    det = load_detector(tmp_path / "legacy.pt")
    assert det.kind == "patchcore" and det.threshold == 1.0


def test_selection_prefers_zero_escape_false_rejects():
    from src.inference.anomaly.benchmark import select_best

    results = [
        {"candidate": "a", "frr_at_recall_100": 0.20, "frr_at_recall_99": 0.1, "image_auroc": 0.99, "latency_ms_mean": 5},
        {"candidate": "b", "frr_at_recall_100": 0.05, "frr_at_recall_99": 0.02, "image_auroc": 0.97, "latency_ms_mean": 40},
    ]
    assert select_best(results)["candidate"] == "b"
    assert select_best(results, latency_budget_ms=10)["candidate"] == "a"


# --------------------------------------------------------- segmentation / typing

def test_synthetic_segmentation_dataset_export(tmp_path):
    import yaml

    from src.process.synthetic_parts import export_segmentation_dataset

    data_yaml = export_segmentation_dataset(tmp_path, n_train=12, n_val=4, width=160, height=120)
    spec = yaml.safe_load(open(data_yaml))
    assert spec["names"] == {0: "short_shot", 1: "sink_mark", 2: "flash", 3: "burn_mark"}
    labels = list((tmp_path / "labels" / "train").glob("*.txt"))
    assert len(labels) == 12
    lines = [ln for f in labels for ln in f.read_text().splitlines()]
    assert lines and all(len(ln.split()) >= 7 and 0 <= int(ln.split()[0]) < 4 for ln in lines)


def test_segmentation_evaluation_picks_zero_escape_threshold():
    from src.inference.segmentation.base import DefectInstance
    from src.inference.segmentation.train import evaluate

    class _Fake:
        def predict(self, image, conf=0.0):
            v = float(image[0, 0, 0]) / 255.0
            m = np.zeros(image.shape[:2], bool)
            m[2:6, 2:6] = True
            return [DefectInstance(0, "d", v, (2, 2, 6, 6), m)] if v >= conf else []

    def item(v, defect):
        img = np.full((8, 8, 3), int(v * 255), np.uint8)
        mask = np.zeros((8, 8), np.uint8)
        if defect:
            mask[2:6, 2:6] = 1
        return img, mask, [0] if defect else []

    items = [item(0.9, True), item(0.7, True), item(0.3, False), item(0.2, False)]
    out = evaluate(_Fake(), items, target_recall=1.0)
    assert out["image_recall"] == 1.0 and out["image_frr"] == 0.0
    assert out["defect_mask_iou_mean"] == pytest.approx(1.0) and out["defect_type_accuracy"] == 1.0


def test_defect_type_classifier_few_shot_and_unknown(tmp_path):
    from src.inference.defect_typing import UNKNOWN, DefectTypeClassifier

    crops, labels = [], []
    for d in ("short_shot", "flash", "sink_mark", "burn_mark"):
        for i in range(8):
            img, mask = render_part(640, 480, d, seed=100 * len(labels) + i, return_mask=True)
            ys, xs = np.where(mask > 0)
            crops.append(img[max(0, ys.min() - 20):ys.max() + 20, max(0, xs.min() - 20):xs.max() + 20])
            labels.append(d)
    clf = DefectTypeClassifier()
    report = clf.fit(crops, labels)
    assert report["cv_accuracy"] > 0.85
    clf.save(tmp_path / "t.joblib")
    loaded = DefectTypeClassifier.load(tmp_path / "t.joblib")
    assert loaded.predict(crops[0])[0] == "short_shot"
    noise = np.random.default_rng(0).integers(0, 255, (80, 80, 3)).astype(np.uint8)
    assert loaded.predict(noise)[0] == UNKNOWN


# ------------------------------------------------------- monitoring / review / export

def test_model_health_flags_score_drift_and_reject_excess():
    from src.inference.anomaly.monitoring import ModelHealthMonitor

    mon = ModelHealthMonitor(calibration={"holdout_mean": 0.5}, threshold=1.0, expected_reject_rate=0.02)
    rng = np.random.default_rng(0)
    for _ in range(100):
        status = mon.update(float(rng.normal(0.5, 0.05)), failed=False)
    assert not status["alerts"] and status["score_drift"] == pytest.approx(1.0, abs=0.1)
    for _ in range(300):  # a full window after the change (one-off outliers must not alarm)
        status = mon.update(float(rng.normal(0.9, 0.05)), failed=False)  # lighting changed
    assert any("imaging conditions" in a for a in status["alerts"])
    mon2 = ModelHealthMonitor(calibration={}, threshold=1.0, expected_reject_rate=0.02)
    for i in range(100):
        status = mon2.update(None, failed=i % 5 == 0)  # 20% rejects vs 2% expected
    assert any("Reject rate" in a for a in status["alerts"])


def test_review_queue_roundtrip_into_audits_and_typing_crops(tmp_path):
    import json

    from src.inference.anomaly.monitoring import ReviewQueue, ingest_reviews

    queue = ReviewQueue(tmp_path / "review", band=(0.8, 1.25))
    assert queue.should_review(1.0) and not queue.should_review(0.3) and not queue.should_review(None)
    meta_path = queue.add("cam", render_part(160, 120, "flash", 1), normalized_score=1.1, verdict="fail", cycle_id="c1")
    meta = json.loads(meta_path.read_text())
    meta["review"] = {"verdict": "fail", "defect_type": "flash", "reviewer": "qa1"}
    meta_path.write_text(json.dumps(meta))

    class _Svc:
        audits = []

        def record_audit(self, cycle_id, verdict, defect_type=None, auditor=""):
            self.audits.append((cycle_id, verdict, defect_type, auditor))

    svc = _Svc()
    stats = ingest_reviews(tmp_path / "review", process_service=svc, crops_dir=tmp_path / "crops")
    assert stats["audits"] == 1 and stats["typing_crops"] == 1 and stats["model_correct"] == 1
    assert svc.audits == [("c1", "fail", "flash", "review:qa1")]
    assert ingest_reviews(tmp_path / "review", process_service=svc) == {}   # idempotent


def test_dinomaly_onnx_export_matches_pytorch(tmp_path):
    from src.inference.anomaly.export import export_dinomaly_onnx
    from src.inference.anomaly.registry import create_detector, load_detector

    good = [render_part(320, 240, None, seed=1000 + i) for i in range(20)]
    det = create_detector("dinomaly", backbone="dinov2reg_vits14", image_size=224, device="cpu")
    det.fit(good, iterations=20, log_every=0)
    meta = export_dinomaly_onnx(det, tmp_path / "m.onnx")
    assert meta["parity_max_abs_error"] < 1e-3
    onnx_det = load_detector(tmp_path / "m.onnx")
    img = render_part(320, 240, "flash", seed=9)
    assert onnx_det.predict(img).score == pytest.approx(det.predict(img).score, rel=1e-3)
