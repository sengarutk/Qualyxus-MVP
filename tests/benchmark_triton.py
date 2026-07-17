from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.config.config import Config
from src.inference.postprocess import cpu_normalize_frame, gpu_normalize_frame, make_preprocessor


def _load_sample_frame() -> np.ndarray:
    cfg = Config.load()
    sample_dir = Path(cfg.get("paths", "data_dir", default="data")) / "samples"
    for ext in ("*.jpg", "*.png", "*.jpeg"):
        files = sorted(sample_dir.glob(ext))
        if files:
            img = cv2.imread(str(files[0]))
            if img is not None:
                return img
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


def _measure(fn, iterations: int = 30) -> tuple[float, float]:
    start = time.perf_counter()
    for _ in range(iterations):
        _ = fn()
    total = time.perf_counter() - start
    latency_ms = (total / iterations) * 1000.0
    fps = iterations / total if total > 0 else 0.0
    return latency_ms, fps


def main() -> None:
    cfg = Config.load()
    frame = _load_sample_frame()
    accel_cfg = cfg.get("hardware_acceleration", default={}) or {}
    precision = str(accel_cfg.get("precision", "fp16"))
    preprocessor = make_preprocessor(use_triton=True, precision=precision)

    cpu_latency, cpu_fps = _measure(lambda: cpu_normalize_frame(frame))
    print(f"CPU/OpenCV normalization: {cpu_latency:.3f} ms/frame | {cpu_fps:.2f} FPS")

    if torch.cuda.is_available():
        gpu_latency, gpu_fps = _measure(lambda: gpu_normalize_frame(frame, precision=precision))
        print(f"PyTorch GPU normalization: {gpu_latency:.3f} ms/frame | {gpu_fps:.2f} FPS")
    else:
        gpu_latency, gpu_fps = 0.0, 0.0
        print("PyTorch GPU normalization: unavailable (CUDA not detected)")

    if preprocessor is not None:
        triton_latency, triton_fps = _measure(lambda: preprocessor.process(frame, device="cuda"))
        print(f"Triton kernel normalization: {triton_latency:.3f} ms/frame | {triton_fps:.2f} FPS")
        if gpu_latency > 0:
            speedup = gpu_latency / max(triton_latency, 1e-9)
            print(f"Triton kernel provides {speedup:.2f}x speedup over standard GPU path")
    else:
        print("Triton kernel normalization: unavailable (falling back to portable CPU/GPU paths)")


if __name__ == "__main__":
    main()