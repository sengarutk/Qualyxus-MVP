"""Model-health monitoring and the human review queue.

A good-parts-only model is only as good as its picture of "normal". A moved camera, a
new light, a dirty lens or a new material lot shifts every score; the line keeps running
while the model silently drifts into rejecting good parts or missing bad ones. Two signals
catch this without any labels:

* score drift - the median normalized score of recently *passed* parts versus the median
  seen on held-out good parts at calibration time;
* reject-rate excess - the observed reject rate versus what the process normally produces,
  tested with a one-sided binomial test so a single bad hour is not an alarm.

Parts whose score falls in an uncertainty band around the threshold are saved, with a
sidecar JSON, to a review queue. Reviewed labels become audits (ground truth for the
process model and the false-reject/escape KPIs) and tagged crops for defect typing.
"""
from __future__ import annotations

import collections
import json
import logging
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("QualyxusModelHealth")


def _binomial_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p) (exact, small n)."""
    if k <= 0:
        return 1.0
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


class ModelHealthMonitor:
    def __init__(self, *, calibration: dict[str, Any] | None, threshold: float | None, window: int = 300,
                 drift_ratio: float = 1.5, expected_reject_rate: float = 0.05, alpha: float = 1e-3):
        calibration = calibration or {}
        baseline = calibration.get("holdout_mean")
        self.baseline_normalized = float(baseline) / threshold if baseline and threshold else None
        self.window = int(window)
        self.drift_ratio = float(drift_ratio)
        self.expected_reject_rate = float(expected_reject_rate)
        self.alpha = float(alpha)
        self._passed = collections.deque(maxlen=self.window)
        self._verdicts = collections.deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._last_alert = 0.0

    def update(self, normalized_score: float | None, failed: bool) -> dict[str, Any]:
        with self._lock:
            self._verdicts.append(bool(failed))
            if normalized_score is not None and not failed:
                self._passed.append(float(normalized_score))
            return self._status()

    def _status(self) -> dict[str, Any]:
        n = len(self._verdicts)
        rejects = int(sum(self._verdicts))
        status: dict[str, Any] = {"parts": n, "reject_rate": round(rejects / n, 4) if n else None,
                                  "score_drift": None, "alerts": []}
        if self.baseline_normalized and len(self._passed) >= 30:
            ratio = float(np.median(self._passed)) / self.baseline_normalized
            status["score_drift"] = round(ratio, 3)
            if ratio > self.drift_ratio:
                status["alerts"].append(
                    f"Passed-part scores are {ratio:.2f}x the calibration level: imaging conditions changed "
                    "(camera position, lighting, lens, material). Re-validate or recalibrate the model.")
        if n >= 50:
            p_value = _binomial_sf(rejects, n, self.expected_reject_rate)
            status["reject_rate_p_value"] = round(p_value, 6)
            if p_value < self.alpha:
                status["alerts"].append(
                    f"Reject rate {rejects / n:.1%} is far above the expected {self.expected_reject_rate:.1%} "
                    f"(p={p_value:.1e}): a real process problem or a stale model - check the review queue.")
        now = time.time()
        if status["alerts"] and now - self._last_alert > 300:
            self._last_alert = now
            for alert in status["alerts"]:
                logger.warning("[ModelHealth] %s", alert)
        return status


class ReviewQueue:
    """Borderline parts saved for operator review (active learning)."""

    def __init__(self, directory: str | Path, *, band: tuple[float, float] = (0.8, 1.25), max_items: int = 5000):
        self.directory = Path(directory)
        self.band = (float(band[0]), float(band[1]))
        self.max_items = int(max_items)

    def should_review(self, normalized_score: float | None) -> bool:
        return normalized_score is not None and self.band[0] <= normalized_score <= self.band[1]

    def add(self, camera_id: str, image: np.ndarray, *, normalized_score: float | None, verdict: str,
            cycle_id: str | None = None, event_id: str | None = None, overlay: np.ndarray | None = None,
            bbox: tuple[int, int, int, int] | None = None, predicted_type: str | None = None,
            image_sha: str | None = None, recipe: str | None = None) -> Path | None:
        target = self.directory / camera_id
        target.mkdir(parents=True, exist_ok=True)
        if len(list(target.glob("*.json"))) >= self.max_items:
            return None
        item_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        cv2.imwrite(str(target / f"{item_id}.png"), image)
        if overlay is not None:
            cv2.imwrite(str(target / f"{item_id}_overlay.png"), overlay)
        meta = {"id": item_id, "camera_id": camera_id, "ts": time.time(), "normalized_score": normalized_score,
                "model_verdict": verdict, "cycle_id": cycle_id, "event_id": event_id,
                "bbox": list(bbox) if bbox else None, "predicted_type": predicted_type,
                "image_sha": image_sha, "recipe": recipe,
                "review": {"verdict": None, "defect_type": None, "reviewer": None}}
        path = target / f"{item_id}.json"
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return path


def ingest_reviews(directory: str | Path, *, process_service=None, crops_dir: str | Path | None = None) -> dict[str, int]:
    """Turn reviewed queue items into audits (process ground truth) and typing crops."""
    stats = collections.Counter()
    for meta_path in sorted(Path(directory).rglob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        review = meta.get("review") or {}
        verdict = review.get("verdict")
        if verdict not in ("pass", "fail") or meta.get("ingested"):
            continue
        if process_service is not None and meta.get("cycle_id"):
            process_service.record_audit(meta["cycle_id"], verdict, defect_type=review.get("defect_type"),
                                         auditor=f"review:{review.get('reviewer') or 'operator'}")
            stats["audits"] += 1
        if crops_dir and verdict == "fail" and review.get("defect_type"):
            img = cv2.imread(str(meta_path.with_suffix(".png")))
            if img is not None:
                out = Path(crops_dir) / str(review["defect_type"])
                out.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out / f"{meta['id']}.png"), img)
                stats["typing_crops"] += 1
        stats[f"model_{'correct' if verdict == meta.get('model_verdict') else 'wrong'}"] += 1
        meta["ingested"] = time.time()
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return dict(stats)
