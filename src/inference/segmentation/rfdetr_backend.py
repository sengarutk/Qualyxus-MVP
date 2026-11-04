"""RF-DETR segmentation backend (Apache-2.0, default)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.inference.segmentation.base import ARCHITECTURES, DefectInstance, DefectSegmenter

logger = logging.getLogger("QualyxusRFDETR")


def _variant(arch: str):
    import rfdetr

    spec = ARCHITECTURES[arch]
    if spec["backend"] != "rfdetr":
        raise ValueError(f"{arch} is not an RF-DETR architecture")
    return getattr(rfdetr, spec["class"])


def train_rfdetr(arch: str, dataset_dir: str | Path, output_dir: str | Path, *, epochs: int = 50,
                 batch_size: int = 4, grad_accum_steps: int = 4, lr: float = 1e-4,
                 resolution: int | None = None, device: str | None = None, **extra: Any) -> dict[str, Any]:
    """Fine-tune from COCO-pretrained weights on a YOLO-format dataset (data.yaml)."""
    cls = _variant(arch)
    model = cls(**({"resolution": resolution} if resolution else {}))
    kwargs: dict[str, Any] = dict(
        dataset_dir=str(dataset_dir), dataset_file="yolo", output_dir=str(output_dir),
        epochs=epochs, batch_size=batch_size, grad_accum_steps=grad_accum_steps, lr=lr, **extra,
    )
    if device:
        kwargs["device"] = device
    model.train(**kwargs)
    out = Path(output_dir)
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        if (out / name).exists():
            return {"weights": name, "resolution": int(model.model_config.resolution)}
    raise FileNotFoundError(f"RF-DETR training produced no best checkpoint in {out}")


class RFDETRSegmenter(DefectSegmenter):
    def __init__(self, card: dict[str, Any], model_dir: Path):
        super().__init__(card, model_dir)
        cls = _variant(card["arch"])
        self.model = cls(
            pretrain_weights=str(self.model_dir / card["weights"]),
            num_classes=len(self.class_names),
            resolution=int(card["resolution"]),
        )
        self._optimize()

    def _optimize(self) -> None:
        """Compiled, FP16-on-GPU inference; optional speed-up that is never required."""
        import torch

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        optimize = getattr(self.model, "inference", None) or getattr(self.model, "optimize_for_inference", None)
        if optimize is None:
            return
        for kwargs in ({"dtype": dtype}, {"dtype": dtype, "compile": False}):
            try:
                optimize(**kwargs)
                return
            except Exception as e:
                logger.debug("[RF-DETR] inference optimisation %s failed: %s", kwargs, e)
        logger.warning("[RF-DETR] running unoptimised (eager FP32) inference")

    def _predict(self, image: np.ndarray, conf: float) -> list[DefectInstance]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        det = self.model.predict(rgb, threshold=conf, include_source_image=False)
        out: list[DefectInstance] = []
        masks = getattr(det, "mask", None)
        for i in range(len(det)):
            cid = int(det.class_id[i]) if det.class_id is not None else 0
            x1, y1, x2, y2 = (int(round(v)) for v in det.xyxy[i])
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            out.append(DefectInstance(cid, name, float(det.confidence[i]), (x1, y1, x2, y2),
                                      None if masks is None else np.asarray(masks[i], dtype=bool)))
        return out
