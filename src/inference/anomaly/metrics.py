"""Industrial anomaly-detection metrics.

Image level: AUROC, average precision, F1-max, and the operating metric plants actually
negotiate on - the false-reject rate when the threshold is set so that no (or at most 1%
of) defective parts escape (``frr_at_recall``).

Pixel level: pixel AUROC and AU-PRO (per-region overlap integrated up to a 30% false
positive rate, the MVTec standard), which - unlike pixel AUROC - weights small defects as
much as large ones.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def image_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=np.float64)
    out: dict[str, float] = {"n": int(y.size), "n_defective": int(y.sum())}
    if len(np.unique(y)) < 2:
        return out
    out["image_auroc"] = float(roc_auc_score(y, s))
    out["image_ap"] = float(average_precision_score(y, s))
    precision, recall, _ = precision_recall_curve(y, s)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    out["image_f1_max"] = float(np.max(f1))
    for r in (1.0, 0.99, 0.95):
        out[f"frr_at_recall_{int(r * 100)}"] = frr_at_recall(y, s, r)
    return out


def frr_at_recall(labels: np.ndarray, scores: np.ndarray, recall: float) -> float:
    """Fraction of good parts rejected at the loosest threshold that still catches
    ``recall`` of the defective parts."""
    defect_scores = np.sort(scores[labels == 1])
    good_scores = scores[labels == 0]
    if defect_scores.size == 0 or good_scores.size == 0:
        return float("nan")
    allowed_misses = int(np.floor((1.0 - recall) * defect_scores.size + 1e-9))
    threshold = defect_scores[allowed_misses]  # parts scoring >= threshold are rejected
    return float(np.mean(good_scores >= threshold))


def _resize_maps(maps: Sequence[np.ndarray], masks: Sequence[np.ndarray], size: int):
    rm, rg = [], []
    for amap, gt in zip(maps, masks):
        rm.append(cv2.resize(amap.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR))
        rg.append(cv2.resize((gt > 0).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST))
    return np.stack(rm), np.stack(rg).astype(bool)


def pixel_metrics(maps: Sequence[np.ndarray], masks: Sequence[np.ndarray], *, size: int = 256,
                  fpr_limit: float = 0.3, steps: int = 200) -> dict[str, float]:
    if not maps:
        return {}
    amaps, gts = _resize_maps(maps, masks, size)
    out: dict[str, float] = {}
    if gts.any() and (~gts).any():
        flat_s, flat_y = amaps.ravel(), gts.ravel()
        if flat_s.size > 4_000_000:  # subsample for speed; AUROC is unbiased under uniform sampling
            idx = np.random.default_rng(0).choice(flat_s.size, 4_000_000, replace=False)
            flat_s, flat_y = flat_s[idx], flat_y[idx]
        out["pixel_auroc"] = float(roc_auc_score(flat_y, flat_s))
        out["pixel_aupro"] = aupro(amaps, gts, fpr_limit=fpr_limit, steps=steps)
    return out


def aupro(amaps: np.ndarray, gts: np.ndarray, *, fpr_limit: float = 0.3, steps: int = 200) -> float:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0, 1]."""
    regions = []  # (image index, boolean mask of one connected defect region)
    for i, gt in enumerate(gts):
        n, labels = cv2.connectedComponents(gt.astype(np.uint8), connectivity=8)
        for r in range(1, n):
            regions.append((i, labels == r))
    if not regions:
        return float("nan")
    normal = ~gts
    n_normal = int(normal.sum())
    lo, hi = float(amaps.min()), float(amaps.max())
    thresholds = np.linspace(hi, lo, steps)
    fprs, pros = [], []
    for t in thresholds:
        pred = amaps >= t
        fprs.append(float(np.logical_and(pred, normal).sum()) / max(n_normal, 1))
        pros.append(float(np.mean([pred[i][region].mean() for i, region in regions])))
        if fprs[-1] > fpr_limit:
            break
    fprs, pros = np.asarray(fprs), np.asarray(pros)
    if fprs[-1] > fpr_limit and len(fprs) > 1:
        # Linear interpolation to land exactly on the FPR limit.
        f0, f1, p0, p1 = fprs[-2], fprs[-1], pros[-2], pros[-1]
        pros[-1] = p0 + (p1 - p0) * (fpr_limit - f0) / max(f1 - f0, 1e-12)
        fprs[-1] = fpr_limit
    return float(np.trapezoid(pros, fprs) / fpr_limit)
