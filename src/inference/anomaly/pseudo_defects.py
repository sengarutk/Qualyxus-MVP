"""Pseudo-defects for validating a good-parts-only recipe before real defects exist.

Two standard generators from the anomaly-detection literature:

* CutPaste (Li et al., CVPR 2021) - a patch cut from a good image pasted elsewhere, with
  random rotation/scale: structural, local anomalies (misplaced or extra material).
* DRAEM-style texture blending (Zavrtanik et al., ICCV 2021) - a smooth random (Perlin-like)
  blob mask filled with a foreign texture: surface anomalies (stains, scratches, burns).

Metrics measured on these are a *proxy*: they show a model is sensitive to local
deviations, not that it will catch a specific real defect. Reports flag them as such.
"""
from __future__ import annotations

import cv2
import numpy as np


def _smooth_blob_mask(h: int, w: int, rng: np.random.Generator, coverage=(0.004, 0.03)) -> np.ndarray:
    """Random smooth blob covering a small fraction of the image."""
    for _ in range(20):
        scale = int(rng.integers(4, 9))
        noise = rng.random((scale, scale)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        thresh = np.quantile(noise, 1.0 - rng.uniform(*coverage))
        mask = (noise >= thresh).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            return (labels == keep).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (w // 2, h // 2), max(3, min(h, w) // 20), 1, -1)
    return mask


def cutpaste(image: np.ndarray, rng: np.random.Generator, source: np.ndarray | None = None):
    src = source if source is not None else image
    h, w = image.shape[:2]
    ph = int(h * rng.uniform(0.05, 0.15))
    pw = int(w * rng.uniform(0.05, 0.15))
    sy, sx = int(rng.integers(0, max(1, src.shape[0] - ph))), int(rng.integers(0, max(1, src.shape[1] - pw)))
    patch = src[sy:sy + ph, sx:sx + pw].copy()
    patch = cv2.rotate(patch, [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][int(rng.integers(3))]) \
        if rng.random() < 0.5 else patch
    ph, pw = patch.shape[:2]
    ty, tx = int(rng.integers(0, max(1, h - ph))), int(rng.integers(0, max(1, w - pw)))
    out = image.copy()
    out[ty:ty + ph, tx:tx + pw] = patch[: h - ty, : w - tx]
    mask = np.zeros((h, w), np.uint8)
    mask[ty:ty + ph, tx:tx + pw] = 1
    return out, mask


def texture_blend(image: np.ndarray, rng: np.random.Generator):
    h, w = image.shape[:2]
    mask = _smooth_blob_mask(h, w, rng)
    texture = rng.integers(0, 255, (8, 8, 3)).astype(np.uint8)
    texture = cv2.resize(texture, (w, h), interpolation=cv2.INTER_CUBIC)
    beta = rng.uniform(0.3, 0.8)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2)[..., None] * beta
    soft[soft < 0.02] = 0.0  # no invisible tails outside the labelled region
    out = (image.astype(np.float32) * (1 - soft) + texture.astype(np.float32) * soft).clip(0, 255).astype(np.uint8)
    # Ground truth = every pixel the blend actually altered (feathered edge included).
    return out, (soft[..., 0] > 0).astype(np.uint8)


def make_pseudo_defect(image: np.ndarray, rng: np.random.Generator, source_pool=None):
    if rng.random() < 0.5:
        source = source_pool[int(rng.integers(len(source_pool)))] if source_pool else None
        if source is not None and source.shape[:2] != image.shape[:2]:
            source = cv2.resize(source, (image.shape[1], image.shape[0]))
        return cutpaste(image, rng, source)
    return texture_blend(image, rng)
