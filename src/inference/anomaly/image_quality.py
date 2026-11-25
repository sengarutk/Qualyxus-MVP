"""Tell a bad image from a bad part.

An anomaly model flags anything unusual, so a defocused lens, a failed lamp or a camera that
was knocked makes every good part look defective. What good images look like (sharpness, noise,
brightness, the background around the part, compression) is stored with the model at
calibration. A frame outside that is an imaging fault: the part is reported as not inspected
and someone fixes the camera, rather than good parts being scrapped.

Light that changes only on the part itself (a soft shadow or reflection inside its outline)
cannot be told from a defect this way; it is still rejected, never passed.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import cv2
import numpy as np

MIN_SHARPNESS_RATIO = 0.5
MAX_NOISE_RATIO = 2.0
MIN_NOISE_RATIO = 0.5
MAX_BLOCK_RATIO = 1.4
LEVEL_MARGIN = 0.10            # how far brightness may leave the range seen in good images
EDGE_SEGMENTS = 8              # per side of the frame
FULL_PIXELS = 2_000_000        # above this, sharpness and noise are measured on tiles at full detail
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


def _tiles(frame: np.ndarray, grid: int = 5, size: int = 256) -> list[np.ndarray]:
    """Grey full-detail tiles spread over a large frame; the whole frame when it is small."""
    h, w = frame.shape[:2]
    if h * w <= FULL_PIXELS:
        return [_gray(frame)]
    ys = np.linspace(0, h - size, grid).astype(int)
    xs = np.linspace(0, w - size, grid).astype(int)
    return [_gray(frame[y:y + size, x:x + size]) for y in ys for x in xs]


def sharpness(frame: np.ndarray) -> float:
    responses = [cv2.Laplacian(t, cv2.CV_32F) for t in _tiles(frame)]
    return float(np.concatenate([r.ravel() for r in responses]).astype(np.float64).var())


def noise_level(frame: np.ndarray) -> float:
    """Sensor noise sigma (Immerkaer's estimator), which ignores image structure."""
    total, pixels = 0.0, 0
    for tile in _tiles(frame):
        h, w = tile.shape[:2]
        total += float(np.abs(cv2.filter2D(tile.astype(np.float32), cv2.CV_32F, _NOISE_KERNEL)).sum())
        pixels += (w - 2) * (h - 2)
    return total * math.sqrt(math.pi / 2.0) / (6.0 * max(pixels, 1))


def blockiness(frame: np.ndarray) -> float:
    """Steps on the 8-pixel grid relative to all others: 1 for a clean image, higher after JPEG."""
    on, off = [], []
    for tile in _tiles(frame):
        if min(tile.shape[:2]) < 16:
            continue
        for axis in (0, 1):
            steps = np.abs(np.diff(tile.astype(np.float32), axis=axis)).mean(axis=1 - axis)
            grid = np.arange(steps.size) % 8 == 7
            on.append(float(np.median(steps[grid])))
            off.append(float(np.median(steps[~grid])))
    # An offset keeps noise-free images (both near zero) from looking compressed.
    return (float(np.mean(on)) + 0.25) / (float(np.mean(off)) + 0.25) if on else 1.0


def light_features(frame: np.ndarray) -> dict[str, Any]:
    """Brightness of the background all round the part, of the part, and clipping."""
    step = max(1, max(frame.shape[:2]) // 640)
    small = _gray(frame[::step, ::step]).astype(np.float32)
    small = cv2.GaussianBlur(small, (3, 3), 0)
    h, w = small.shape
    k = max(3, int(round(0.05 * min(h, w))))
    sides = (small[:k], small[-k:], small[:, :k].T, small[:, -k:].T)
    segments, texture = [], []
    for side in sides:
        for part in np.array_split(side, EDGE_SEGMENTS, axis=1):
            segments.append(float(np.median(part)))
            texture.append(float(part.std()))
    # The middle of the part's own brightness: a burn, a missing corner or flash does not move it.
    part = small[np.abs(small - float(np.median(segments))) > 12.0]
    return {"edge": segments, "edge_texture": float(np.median(texture)),
            "bright": float(np.median(part)) if part.size >= 0.01 * small.size else None, "clipped": float((small >= 250).mean())}


def image_quality_baseline(images: Sequence[np.ndarray]) -> dict[str, Any]:
    lights = [light_features(i) for i in images]
    bright = [f["bright"] for f in lights if f["bright"] is not None]
    edges = np.array([f["edge"] for f in lights])
    return {"sharpness": float(np.median([sharpness(i) for i in images])),
            "noise": float(np.median([noise_level(i) for i in images])),
            "blockiness": float(np.median([blockiness(i) for i in images])),
            "edge_low": np.round(edges.min(axis=0), 2).tolist(), "edge_high": np.round(edges.max(axis=0), 2).tolist(),
            "edge_texture": float(np.max([f["edge_texture"] for f in lights])),
            "bright_low": min(bright) if len(bright) == len(lights) else None,
            "bright_high": max(bright) if len(bright) == len(lights) else None,
            "clipped": float(np.max([f["clipped"] for f in lights]))}


def _margin(level: float) -> float:
    return max(LEVEL_MARGIN * level, 3.0)


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
    if "edge_low" not in baseline:       # calibrated before light was checked
        return None
    light = light_features(frame)
    if light["clipped"] > float(baseline["clipped"]) + 0.02:
        return f"{light['clipped']:.0%} of the image is overexposed (glare)"
    edge, low, high = (np.asarray(v, dtype=np.float64) for v in (light["edge"], baseline["edge_low"], baseline["edge_high"]))
    under = edge < low - np.maximum(LEVEL_MARGIN * low, 3.0)
    over = edge > high + np.maximum(LEVEL_MARGIN * high, 3.0)
    wrong = int((under | over).sum())
    if wrong == edge.size:
        return f"image is too {'dark' if under.all() else 'bright' if over.all() else 'unevenly lit'}"
    if wrong:
        return (f"{wrong} of {edge.size} background areas changed (uneven light, something in front of the camera, "
                "or the part is out of position)")
    if light["edge_texture"] > 2.0 * float(baseline["edge_texture"]) + 2.0:
        return "the background has a pattern the model has not seen"
    if light["bright"] is not None and baseline.get("bright_low") is not None:
        low, high = float(baseline["bright_low"]), float(baseline["bright_high"])
        if light["bright"] < low - 1.5 * _margin(low):
            return f"part is too dark ({light['bright']:.0f}, normally above {low:.0f})"
        if light["bright"] > high + 1.5 * _margin(high):
            return f"part is too bright ({light['bright']:.0f}, normally below {high:.0f})"
    if noise_ratio < MIN_NOISE_RATIO:
        return f"image noise is only {noise_ratio:.2f}x normal (compression or smoothing has removed detail)"
    block_ratio = blockiness(frame) / max(float(baseline["blockiness"]), 1e-6)
    if block_ratio > MAX_BLOCK_RATIO:
        return f"image is compressed ({block_ratio:.1f}x normal block edges)"
    return None
