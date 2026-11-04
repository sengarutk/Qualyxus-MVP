"""Common interface for good-parts-only anomaly detectors."""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from src.inference.anomaly.calibration import conformal_threshold
from src.inference.anomaly.preprocess import Preprocessor

logger = logging.getLogger("QualyxusAnomaly")

FORMAT = "qualyxus-anomaly-v2"


@dataclass
class AnomalyResult:
    score: float               # raw image anomaly score
    normalized_score: float    # score / threshold; > 1.0 means anomalous
    is_anomalous: bool
    anomaly_map: np.ndarray    # HxW float32 at the input image resolution
    bbox: tuple[int, int, int, int] | None  # x1, y1, x2, y2 of the dominant anomalous region


class AnomalyDetector(ABC):
    kind = "base"

    def __init__(self, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold: float | None = None
        self.calibration: dict[str, Any] = {}
        self.preprocessor = Preprocessor()
        self._lock = threading.Lock()

    def configure_preprocessing(self, *, register: bool = False, tiles: tuple[int, int] | None = None,
                                overlap: float = 0.15) -> "AnomalyDetector":
        """Part registration and/or tiling in front of the model (set before ``fit``)."""
        self.preprocessor = Preprocessor(register=register, tiles=tuple(tiles) if tiles else None, overlap=overlap)
        return self

    # -------------------------------------------------------------- training

    @abstractmethod
    def _train(self, images: Sequence[np.ndarray], **kwargs) -> dict[str, Any]:
        """Learn the model of normality from good images."""

    @abstractmethod
    def score_map(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """(image score, anomaly map at the input resolution) for one BGR image."""

    def score_batch(self, images: Sequence[np.ndarray]) -> list[tuple[float, np.ndarray]]:
        return [self.score_map(img) for img in images]

    def infer(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        """Image score and frame-sized anomaly map, through registration / tiling when configured."""
        if not self.preprocessor.active:
            return self.score_map(image)
        prepared = self.preprocessor.prepare(image)
        return self.preprocessor.combine(prepared, self.score_batch([v.image for v in prepared.views]))

    def infer_batch(self, images: Sequence[np.ndarray]) -> list[tuple[float, np.ndarray]]:
        if not self.preprocessor.active:
            return self.score_batch(images)
        return [self.infer(img) for img in images]

    def _holdout_scores(self, holdout: Sequence[np.ndarray]) -> list[float]:
        """Image scores of held-out good images used for threshold calibration."""
        return [s for s, _ in self.infer_batch(holdout)]

    def fit(self, good_images: Sequence[np.ndarray], *, holdout_fraction: float = 0.2,
            target_frr: float = 0.01, confidence: float | None = None, seed: int = 0,
            **train_kwargs) -> dict[str, Any]:
        """Train on good images, then calibrate the threshold on held-out good images."""
        images = list(good_images)
        if len(images) < 10:
            raise ValueError(f"Need at least 10 good images (got {len(images)})")
        order = np.random.default_rng(seed).permutation(len(images))
        n_hold = max(3, int(round(len(images) * holdout_fraction)))
        holdout = [images[i] for i in order[:n_hold]]
        train = [images[i] for i in order[n_hold:]]
        prep_info = self.preprocessor.fit(train)
        views = [v for img in train for v in self.preprocessor.views(img)] if self.preprocessor.active else train
        info = self._train(views, **train_kwargs)
        self.calibrate(self._holdout_scores(holdout), target_frr=target_frr, confidence=confidence)
        self.calibration.update({"n_train": len(train), **(info or {})})
        if self.preprocessor.active:
            self.calibration["preprocess"] = {"register": self.preprocessor.register,
                                              "tiles": self.preprocessor.tiles, **prep_info}
        return dict(self.calibration)

    def calibrate(self, good_scores: Sequence[float], target_frr: float = 0.01,
                  confidence: float | None = None) -> dict[str, Any]:
        result = conformal_threshold(good_scores, target_frr=target_frr, confidence=confidence)
        self.threshold = float(result["threshold"])
        self.calibration = dict(result)
        if not result["frr_guaranteed"]:
            logger.warning(
                "[Anomaly] %d held-out good images cannot guarantee FRR <= %.3g (need >= %d); "
                "using max good score + margin.", result["n_holdout"], target_frr, result["min_holdout_for_guarantee"],
            )
        return result

    # ------------------------------------------------------------- inference

    def predict(self, image: np.ndarray) -> AnomalyResult:
        if self.threshold is None:
            raise RuntimeError(f"{self.kind} detector has no calibrated threshold")
        with self._lock:
            score, amap = self.infer(image)
        normalized = score / self.threshold if self.threshold > 0 else float("inf")
        bbox = None
        if normalized > 1.0:
            # Localise against the *pixel* distribution of the map: the strongest region.
            level = max(self.threshold, float(amap.max()) * 0.5)
            mask = (amap >= min(level, float(amap.max()))).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                bbox = (int(x), int(y), int(x + w), int(y + h))
        return AnomalyResult(
            score=round(float(score), 6),
            normalized_score=round(float(normalized), 5),
            is_anomalous=bool(normalized > 1.0),
            anomaly_map=amap,
            bbox=bbox,
        )

    @staticmethod
    def heatmap_overlay(image: np.ndarray, result: AnomalyResult, threshold: float) -> np.ndarray:
        scaled = np.clip(result.anomaly_map / (2.0 * threshold), 0.0, 1.0) if threshold > 0 else result.anomaly_map
        heat = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heat, 0.4, 0)
        if result.bbox:
            x1, y1, x2, y2 = result.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return overlay

    # ----------------------------------------------------------- persistence

    @abstractmethod
    def _state(self) -> dict[str, Any]:
        """Serializable (tensor/primitive only) model state."""

    @classmethod
    @abstractmethod
    def _from_state(cls, state: dict[str, Any], device: str | None) -> "AnomalyDetector":
        ...

    def save(self, path: str | Path) -> None:
        if self.threshold is None:
            raise RuntimeError("Cannot save an uncalibrated detector")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._state()
        torch.save({
            "format": FORMAT,
            "kind": self.kind,
            "state": state,
            "threshold": self.threshold,
            "calibration": self.calibration,
            "preprocess": self.preprocessor.state(),
        }, path)
        # Human-readable model card next to the weights (what is deployed, under which licence).
        import json
        import time

        from src.inference.anomaly.backbones import backbone_license

        backbone = state.get("backbone")
        card = {
            "kind": self.kind,
            "backbone": backbone,
            "backbone_license": backbone_license(backbone) if backbone else None,
            "config": {k: v for k, v in state.items() if isinstance(v, (int, float, str, list, type(None)))},
            "preprocess": {k: v for k, v in self.preprocessor.state().items() if k != "template"},
            "threshold": self.threshold,
            "calibration": self.calibration,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        path.with_suffix(".card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")

    def _restore_common(self, payload: dict[str, Any]) -> None:
        self.threshold = float(payload["threshold"])
        self.calibration = dict(payload.get("calibration") or {})
        self.preprocessor = Preprocessor.from_state(payload.get("preprocess"))
