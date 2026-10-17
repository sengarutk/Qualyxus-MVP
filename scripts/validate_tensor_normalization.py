from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_ROOT_DIR = _Path(__file__).resolve().parents[1]
if str(_ROOT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_ROOT_DIR))


"""Validate GPU-side preprocessing (uint8->float32 + /255) and Ultralytics warnings.

This script exists to validate that our preprocessing path feeds Ultralytics a
contiguous float32 RGB tensor already scaled to 0..1 so Ultralytics does not
emit its internal "Dividing input by 255" / "torch.Tensor inputs" warnings.

Run:
  PYTHONPATH=. .venv/bin/python scripts/validate_tensor_normalization.py
"""

import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from src.inference.postprocess import make_preprocessor, preprocess_for_inference


def _pick_sample_image() -> str:
    # Prefer any recorded live frame if present (CameraManager writes PNG frames).
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        candidates = list(Path("data/live/raw").rglob(pattern))
        if candidates:
            return str(sorted(candidates)[-1])
    # Fallback to repo samples.
    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        samples = list(Path("data/samples").glob(pattern))
        if samples:
            return str(sorted(samples)[0])
    raise FileNotFoundError("No sample images found under data/live/raw or data/samples")


def main() -> int:
    img_path = sys.argv[1] if len(sys.argv) > 1 else _pick_sample_image()
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    # Ultralytics tensor input path requires BCHW with H=W=imgsz by default.
    # Match what CameraManager does (resize to configured width/height), but keep it square
    # here to avoid LoadTensor shape validation errors.
    img = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = "fp16" if device == "cuda" else "fp32"

    pre = make_preprocessor(use_triton=True, precision=precision)
    tensor, used_triton = preprocess_for_inference(
        img,
        device=device,
        use_triton=bool(pre) and device == "cuda",
        precision=precision,
        preprocessor=pre,
    )

    print(f"image={img_path}")
    print(f"device={device} used_triton={used_triton}")
    print(
        "tensor:",
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        "contig=",
        tensor.is_contiguous(),
    )

    weights = "runs/segment/models/checkpoints/severstal_yolov8_seg/weights/best.pt"
    model = YOLO(weights)
    if device == "cuda":
        model.to("cuda")

    # Keep tensor float32; enable FP16 inference via half flag when on CUDA.
    _ = model.predict(tensor, device=device, verbose=False, half=(device == "cuda"), imgsz=640)
    print("predict_ok=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
