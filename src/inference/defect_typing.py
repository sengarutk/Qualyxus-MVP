"""Few-shot defect-type classification from operator-tagged crops.

The anomaly detector says *something is wrong and where*; operators name it. A frozen
DINOv2 encoder turns each defect crop into an embedding (CLS token + mean patch token,
L2-normalised); a regularised logistic regression on top learns the types from a handful
of tagged examples per class. Crops far from every known type - nearest-neighbour cosine
distance beyond the calibrated novelty radius - are reported as ``"unknown"`` instead of
being forced into a wrong class, so new failure modes surface for tagging.

    python -m src.inference.defect_typing --crops data/labels/defect_crops --out models/typing/<recipe>.joblib

``--crops`` holds one sub-folder per defect type.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch

from src.inference import signed_model

logger = logging.getLogger("QualyxusDefectTyping")
UNKNOWN = "unknown"


class _Embedder:
    def __init__(self, backbone: str = "dinov2_vits14", image_size: int = 224):
        from src.inference.anomaly.backbones import ViTBackbone

        self.backbone = ViTBackbone(backbone)
        self.image_size = self.backbone.align_size(image_size)

    @torch.inference_mode()
    def __call__(self, crops: Sequence[np.ndarray], batch: int = 32) -> np.ndarray:
        from src.inference.anomaly.backbones import autocast_dtype, to_tensor_batch

        out = []
        for i in range(0, len(crops), batch):
            x = to_tensor_batch(crops[i:i + batch], self.image_size, self.backbone.device)
            with torch.autocast(device_type=x.device.type, dtype=autocast_dtype(x.device), enabled=x.device.type == "cuda"):
                hidden = self.backbone.model(pixel_values=x).last_hidden_state.float()
            cls_tok = hidden[:, 0]
            patch_mean = hidden[:, self.backbone.num_prefix:].mean(dim=1)
            emb = torch.cat([cls_tok, patch_mean], dim=1)
            out.append(torch.nn.functional.normalize(emb, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2 * self.backbone.embed_dim), np.float32)


@dataclass
class DefectTypeClassifier:
    backbone: str = "dinov2_vits14"
    image_size: int = 224
    min_probability: float = 0.5
    classes: list[str] = field(default_factory=list)
    _clf: Any = None
    _bank: np.ndarray | None = None
    novelty_radius: float = 1.0
    _embedder: Any = None

    def _embed(self, crops):
        if self._embedder is None:
            self._embedder = _Embedder(self.backbone, self.image_size)
        return self._embedder(crops)

    def fit(self, crops: Sequence[np.ndarray], labels: Sequence[str], *, seed: int = 0) -> dict[str, Any]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict

        labels = list(labels)
        self.classes = sorted(set(labels))
        if len(self.classes) < 2:
            raise ValueError("Need at least two defect types to train a type classifier")
        X = self._embed(crops)
        y = np.array([self.classes.index(lbl) for lbl in labels])
        self._clf = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced")
        report: dict[str, Any] = {"n": int(len(y)), "per_class": {c: labels.count(c) for c in self.classes}}
        min_count = min(labels.count(c) for c in self.classes)
        if min_count >= 2:
            folds = min(5, min_count)
            pred = cross_val_predict(self._clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=seed))
            report["cv_accuracy"] = float((pred == y).mean())
        self._clf.fit(X, y)
        self._bank = X
        # Novelty radius: 99th percentile of leave-one-out nearest-neighbour distances.
        sims = X @ X.T
        np.fill_diagonal(sims, -1.0)
        nn_dist = 1.0 - sims.max(axis=1)
        self.novelty_radius = float(np.quantile(nn_dist, 0.99) * 1.25 + 1e-3)
        report["novelty_radius"] = self.novelty_radius
        return report

    def predict(self, crop: np.ndarray) -> tuple[str, float]:
        if self._clf is None:
            raise RuntimeError("Defect type classifier is not trained")
        emb = self._embed([crop])
        nearest = 1.0 - float((self._bank @ emb[0]).max())
        probs = self._clf.predict_proba(emb)[0]
        k = int(np.argmax(probs))
        if nearest > self.novelty_radius or probs[k] < self.min_probability:
            return UNKNOWN, float(probs[k])
        return self.classes[k], float(probs[k])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        signed_model.save({"backbone": self.backbone, "image_size": self.image_size,
                           "min_probability": self.min_probability, "classes": self.classes, "clf": self._clf,
                           "bank": self._bank, "novelty_radius": self.novelty_radius}, path)

    @classmethod
    def load(cls, path: str | Path) -> "DefectTypeClassifier":
        d = signed_model.load(path)
        obj = cls(backbone=d["backbone"], image_size=d["image_size"], min_probability=d["min_probability"],
                  classes=list(d["classes"]))
        obj._clf, obj._bank, obj.novelty_radius = d["clf"], d["bank"], d["novelty_radius"]
        return obj


def main(argv=None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", required=True, help="Folder with one sub-folder of crops per defect type")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backbone", default="dinov2_vits14")
    args = parser.parse_args(argv)
    crops, labels = [], []
    for type_dir in sorted(p for p in Path(args.crops).iterdir() if p.is_dir()):
        for f in sorted(type_dir.iterdir()):
            img = cv2.imread(str(f))
            if img is not None:
                crops.append(img)
                labels.append(type_dir.name)
    clf = DefectTypeClassifier(backbone=args.backbone)
    report = clf.fit(crops, labels)
    clf.save(args.out)
    logger.info("Saved %s: %s", args.out, report)
    return report


if __name__ == "__main__":
    main()
