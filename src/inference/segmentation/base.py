"""Supervised defect segmentation: common types and the model-card loader.

Supervised segmentation complements the good-parts-only anomaly detector once labelled
defect examples exist: it names the defect type and delineates it precisely. Every
trained model is stored as a directory with a ``model_card.json`` describing its backend,
architecture, license, classes, operating threshold and validation metrics.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CARD_NAME = "model_card.json"

ARCHITECTURES: dict[str, dict[str, Any]] = {
    # RF-DETR segmentation (Roboflow, ICLR 2026): DINOv2 backbone, Apache-2.0 incl. weights.
    "rfdetr-seg-nano": {"backend": "rfdetr", "class": "RFDETRSegNano", "license": "Apache-2.0"},
    "rfdetr-seg-small": {"backend": "rfdetr", "class": "RFDETRSegSmall", "license": "Apache-2.0"},
    "rfdetr-seg-medium": {"backend": "rfdetr", "class": "RFDETRSegMedium", "license": "Apache-2.0"},
    "rfdetr-seg-large": {"backend": "rfdetr", "class": "RFDETRSegLarge", "license": "Apache-2.0"},
    # Ultralytics YOLO26 segmentation: AGPL-3.0 - commercial deployment needs an Ultralytics licence.
    "yolo26n-seg": {"backend": "yolo", "weights": "yolo26n-seg.pt", "license": "AGPL-3.0"},
    "yolo26s-seg": {"backend": "yolo", "weights": "yolo26s-seg.pt", "license": "AGPL-3.0"},
    "yolo26m-seg": {"backend": "yolo", "weights": "yolo26m-seg.pt", "license": "AGPL-3.0"},
}
DEFAULT_ARCH = "rfdetr-seg-small"


@dataclass
class DefectInstance:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None  # HxW bool at the input image resolution

    @property
    def area(self) -> int:
        return int(self.mask.sum()) if self.mask is not None else max(0, (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


class DefectSegmenter(ABC):
    def __init__(self, card: dict[str, Any], model_dir: Path):
        self.card = card
        self.model_dir = Path(model_dir)
        self.class_names: list[str] = list(card.get("class_names") or [])
        self.conf_threshold = float(card.get("conf_threshold", 0.5))
        self.min_area_px = int(card.get("min_area_px", 0))

    @abstractmethod
    def _predict(self, image: np.ndarray, conf: float) -> list[DefectInstance]:
        ...

    def predict(self, image: np.ndarray, conf: float | None = None) -> list[DefectInstance]:
        instances = self._predict(image, self.conf_threshold if conf is None else float(conf))
        return [i for i in instances if i.area >= self.min_area_px]


def load_segmenter(model_dir: str | Path) -> DefectSegmenter:
    model_dir = Path(model_dir)
    card = json.loads((model_dir / CARD_NAME).read_text(encoding="utf-8"))
    backend = card.get("backend")
    if backend == "rfdetr":
        from src.inference.segmentation.rfdetr_backend import RFDETRSegmenter

        return RFDETRSegmenter(card, model_dir)
    if backend == "yolo":
        from src.inference.segmentation.yolo_backend import YOLOSegmenter

        return YOLOSegmenter(card, model_dir)
    raise ValueError(f"Unknown segmentation backend '{backend}' in {model_dir}")


def write_card(model_dir: str | Path, card: dict[str, Any]) -> Path:
    path = Path(model_dir) / CARD_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    return path
