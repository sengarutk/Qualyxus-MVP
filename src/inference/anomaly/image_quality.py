"""Tell a bad image from a bad part.

An anomaly model flags anything unusual, so a defocused lens or a failing camera makes every
good part look defective. The sharpness and noise of the good images used for calibration
are stored with the model. A frame far outside them is an imaging fault: the part is reported
as not inspected and someone fixes the camera, rather than good parts being scrapped.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

MIN_SHARPNESS_RATIO = 0.5
MAX_NOISE_RATIO = 2.0
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def sharpness(frame: np.ndarray) -> float:
    return float(cv2.Laplacian(_gray(frame), cv2.CV_64F).var())


def noise_level(frame: np.ndarray) -> float:
    """Sensor noise sigma (Immerkaer's estimator), which ignores image structure."""
    gray = _gray(frame).astype(np.float32)
    h, w = gray.shape[:2]
    response = cv2.filter2D(gray, cv2.CV_32F, _NOISE_KERNEL)
    return float(np.abs(response).sum() * math.sqrt(math.pi / 2.0) / (6.0 * (w - 2) * (h - 2)))


def image_quality_baseline(images: Sequence[np.ndarray]) -> dict[str, float]:
    return {"sharpness": float(np.median([sharpness(i) for i in images])),
            "noise": float(np.median([noise_level(i) for i in images]))}


def image_fault(frame: np.ndarray, baseline: dict[str, Any] | None) -> str | None:
    """Why this frame cannot be judged fairly, or None. No baseline means no check."""
    if not baseline:
        return None
    noise_ratio = noise_level(frame) / max(float(baseline["noise"]), 1e-6)
    if noise_ratio > MAX_NOISE_RATIO:
        return f"image noise is {noise_ratio:.1f}x normal"
    sharpness_ratio = sharpness(frame) / max(float(baseline["sharpness"]), 1e-6)
    if sharpness_ratio < MIN_SHARPNESS_RATIO:
        return f"image sharpness is {sharpness_ratio:.2f}x normal (focus, dirty lens or motion blur)"
    return None
