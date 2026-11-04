"""Ultralytics YOLO segmentation backend (AGPL-3.0 - opt-in).

Ultralytics models and weights are AGPL-3.0: shipping them inside a commercial product
requires either open-sourcing that product or an Ultralytics Enterprise licence. RF-DETR
(Apache-2.0) is the default for that reason; this backend exists for comparison and for
teams that hold an Ultralytics licence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.inference.segmentation.base import ARCHITECTURES, DefectInstance, DefectSegmenter

logger = logging.getLogger("QualyxusYOLOSeg")


def train_yolo(arch: str, data_yaml: str | Path, output_dir: str | Path, *, epochs: int = 50, batch_size: int = 8,
               imgsz: int = 640, device: str | int | None = None, **extra: Any) -> dict[str, Any]:
    from ultralytics import YOLO

    spec = ARCHITECTURES[arch]
    logger.warning("[YOLO] %s is AGPL-3.0; commercial deployment requires an Ultralytics licence.", arch)
    out = Path(output_dir)
    model = YOLO(spec["weights"])
    model.train(data=str(data_yaml), epochs=epochs, batch=batch_size, imgsz=imgsz,
                project=str(out.parent.resolve()), name=out.name, exist_ok=True,
                **({"device": device} if device is not None else {}), **extra)
    best = out / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(best)
    return {"weights": "weights/best.pt", "resolution": imgsz}


class YOLOSegmenter(DefectSegmenter):
    def __init__(self, card: dict[str, Any], model_dir: Path):
        super().__init__(card, model_dir)
        from ultralytics import YOLO

        self.model = YOLO(str(self.model_dir / card["weights"]))
        self.imgsz = int(card.get("resolution", 640))

    def _predict(self, image: np.ndarray, conf: float) -> list[DefectInstance]:
        res = self.model.predict(image, conf=conf, imgsz=self.imgsz, verbose=False, retina_masks=True)[0]
        out: list[DefectInstance] = []
        if res.boxes is None:
            return out
        masks = res.masks.data.cpu().numpy() if res.masks is not None else None
        for i in range(len(res.boxes)):
            cid = int(res.boxes.cls[i])
            x1, y1, x2, y2 = (int(round(float(v))) for v in res.boxes.xyxy[i])
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            mask = None if masks is None else masks[i].astype(bool)
            out.append(DefectInstance(cid, name, float(res.boxes.conf[i]), (x1, y1, x2, y2), mask))
        return out
