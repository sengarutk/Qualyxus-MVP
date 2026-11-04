"""Score-level ensemble of anomaly detectors.

Different detectors fail on different defects: memory-bank PatchCore is strongest on
small local texture defects, reconstruction-based Dinomaly on structural/logical ones
(missing or misplaced features). Fusing their *normalised* scores is a standard way to get
the best of both.

Each member's raw scores live on a different scale, so every member is standardised with
robust statistics (median / IQR) measured on held-out good images. To keep the conformal
threshold guarantee intact, those statistics come from one half of the hold-out set and
the fused threshold is calibrated on the other, disjoint half.
"""
from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from src.inference.anomaly.base import AnomalyDetector

logger = logging.getLogger("QualyxusEnsemble")


class EnsembleDetector(AnomalyDetector):
    kind = "ensemble"

    def __init__(self, members: Sequence[AnomalyDetector], *, weights: Sequence[float] | None = None,
                 device: str | None = None):
        super().__init__(device)
        if len(members) < 2:
            raise ValueError("An ensemble needs at least two members")
        self.members = list(members)
        w = np.asarray(weights if weights is not None else [1.0] * len(self.members), dtype=np.float64)
        self.weights = (w / w.sum()).tolist()
        self.norm: list[tuple[float, float]] = []   # (median, scale) per member

    # -------------------------------------------------------------- training

    def _train(self, images: Sequence[np.ndarray], **kwargs) -> dict[str, Any]:
        info = {}
        for i, member in enumerate(self.members):
            member_kwargs = dict(kwargs)
            if member.kind != "dinomaly":
                member_kwargs = {k: v for k, v in member_kwargs.items() if k in ("batch_size", "seed")}
            info[f"member{i}_{member.kind}"] = member._train(images, **member_kwargs)
        return {"members": info}

    def _holdout_scores(self, holdout: Sequence[np.ndarray]) -> list[float]:
        n_norm = len(holdout) // 2
        norm_set, cal_set = list(holdout[:n_norm]), list(holdout[n_norm:])
        if n_norm < 5:
            raise ValueError("Ensemble calibration needs at least 10 held-out good images")
        self.norm = []
        for member in self.members:
            raw = np.array(self._member_image_scores(member, norm_set), dtype=np.float64)
            q25, med, q75 = np.percentile(raw, [25, 50, 75])
            self.norm.append((float(med), float(max(q75 - q25, 1e-6 * max(abs(med), 1.0)))))
        logger.info("[Ensemble] member normalisation from %d images; threshold from %d", len(norm_set), len(cal_set))
        return [s for s, _ in self.infer_batch(cal_set)]

    def _member_image_scores(self, member: AnomalyDetector, images: Sequence[np.ndarray]) -> list[float]:
        """A member's image-level scores through the ensemble's registration / tiling."""
        if not self.preprocessor.active:
            return [s for s, _ in member.score_batch(images)]
        return [max(s for s, _ in member.score_batch(self.preprocessor.views(img))) for img in images]

    # ------------------------------------------------------------- inference

    def _fuse(self, per_member: Sequence[Sequence[tuple[float, np.ndarray]]]) -> list[tuple[float, np.ndarray]]:
        out = []
        for results in zip(*per_member):
            score, amap = 0.0, None
            for (s, m), (med, scale), w in zip(results, self.norm, self.weights):
                score += w * (s - med) / scale
                z = (m - med) / scale
                amap = w * z if amap is None else amap + w * z
            out.append((float(score), amap.astype(np.float32)))
        return out

    def score_map(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        return self.score_batch([image])[0]

    def score_batch(self, images: Sequence[np.ndarray]) -> list[tuple[float, np.ndarray]]:
        if not self.norm:
            raise RuntimeError("Ensemble is not calibrated")
        return self._fuse([member.score_batch(images) for member in self.members])

    # ----------------------------------------------------------- persistence

    def _state(self) -> dict[str, Any]:
        return {
            "backbone": "+".join(getattr(m, "backbone_name", m.kind) for m in self.members),
            "members": [{"kind": m.kind, "state": m._state()} for m in self.members],
            "weights": list(self.weights),
            "norm": [list(n) for n in self.norm],
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any], device: str | None) -> "EnsembleDetector":
        from src.inference.anomaly.registry import detector_class

        members = [detector_class(m["kind"])._from_state(m["state"], device) for m in state["members"]]
        det = cls(members, weights=state["weights"], device=device)
        det.norm = [tuple(n) for n in state["norm"]]
        return det
