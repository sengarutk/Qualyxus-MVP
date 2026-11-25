"""Is a part in view? Decided on a small copy of the frame, so a 12 MP camera costs what VGA does.

Without a reference the empty station is modelled as smooth light: a surface fitted to the edge
of the frame. Uneven lighting is then not a part, and a part is whatever stands out from it.
With a reference image of the empty station (``presence.reference``) the frame is compared with
that instead. Use one when the background has a pattern, or the part reaches the frame edge:

    python -m src.camera.presence learn --source rtsp://... --out data/templates/empty_top.png

The reference follows slow changes of light, but only where no part is.
"""
from __future__ import annotations

import argparse
import functools
import logging
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("QualyxusPresence")

LONG_SIDE = 192
MIN_CONTRAST = 4.0          # grey levels, after averaging down
MIN_AREA = 0.004            # share of the frame
REFERENCE_RATE = 0.01


def small_gray(frame: np.ndarray, long_side: int = LONG_SIDE) -> np.ndarray:
    """Float grey copy with the long side near ``long_side``; sampled first, so size hardly matters."""
    step = max(1, max(frame.shape[:2]) // (long_side * 2))
    small = frame[::step, ::step]
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.shape[2] == 3 else small[:, :, 0]
    scale = long_side / max(small.shape[:2])
    if scale < 1.0:
        small = cv2.resize(small, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(small.astype(np.float32), (3, 3), 0)


@functools.lru_cache(maxsize=16)
def _terms(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = x / max(w - 1, 1) - 0.5, y / max(h - 1, 1) - 0.5
    return np.stack([np.ones_like(x), x, y, x * y, x * x, y * y], axis=-1).reshape(-1, 6)


@functools.lru_cache(maxsize=16)
def border_mask(h: int, w: int, fraction: float = 0.06) -> np.ndarray:
    k = max(2, int(round(fraction * min(h, w))))
    mask = np.ones((h, w), bool)
    mask[k:-k, k:-k] = False
    return mask


@functools.lru_cache(maxsize=16)
def _edge(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """About 600 points of the frame's edge and their terms."""
    index = np.flatnonzero(border_mask(h, w))
    index = index[::max(1, index.size // 600)]
    return index, _terms(h, w)[index].astype(np.float64)


def light_surface(small: np.ndarray) -> tuple[np.ndarray, float]:
    """(smooth light over the frame, how far the edge scatters around it), fitted to the edge only."""
    h, w = small.shape
    index, a = _edge(h, w)
    b = small.ravel()[index].astype(np.float64)
    keep = np.ones(b.size, bool)
    coef, spread = np.zeros(6), 0.0
    for _ in range(4):          # whatever touches the edge must not bend the surface
        # Small normal equations: cheap, and no thread pool is woken for every frame of every camera.
        ak, bk = a[keep], b[keep]
        coef = np.linalg.solve(ak.T @ ak + 1e-6 * np.eye(6), ak.T @ bk)
        residual = np.abs(b - a @ coef)
        spread = 1.4826 * float(np.median(residual))
        keep = residual <= max(3.0 * spread, 1.0)
        if keep.sum() < 24:
            break
    return (_terms(h, w) @ coef.astype(np.float32)).reshape(h, w), spread


def _largest(mask: np.ndarray) -> tuple[float, tuple[int, int, int, int] | None]:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return 0.0, None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, area = (int(v) for v in stats[best])
    return area / mask.size, (x, y, w, h)


class PresenceDetector:
    def __init__(self, *, min_contrast: float = MIN_CONTRAST, min_area: float = MIN_AREA,
                 reference: np.ndarray | None = None):
        self.min_contrast, self.min_area = float(min_contrast), float(min_area)
        self._reference = None if reference is None else small_gray(reference)
        self._start = None if self._reference is None else self._reference.copy()
        self.box: tuple[float, float, float, float] | None = None      # where the part is, 0..1

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "PresenceDetector":
        cfg = cfg or {}
        reference = None
        if cfg.get("reference"):
            reference = cv2.imread(str(cfg["reference"]))
            if reference is None:
                raise ValueError(f"presence.reference {cfg['reference']} cannot be read")
        return cls(min_contrast=float(cfg.get("min_contrast", MIN_CONTRAST)),
                   min_area=float(cfg.get("min_area", MIN_AREA)), reference=reference)

    def __call__(self, frame: np.ndarray) -> bool:
        if os.getenv("QUALYXUS_SAVE_BLANK_FRAMES", "0").strip().lower() in {"1", "true", "yes"}:
            return True
        if frame is None or frame.size == 0:
            return False
        small = small_gray(frame)
        if self._reference is not None and self._reference.shape == small.shape:
            # One gain for the whole frame: a lamp that dims is not a part.
            edge = border_mask(*small.shape)
            gain = float(np.median(small[edge] / np.maximum(self._reference[edge], 1.0)))
            gain = float(np.clip(gain, 0.7, 1.4))
            diff = np.abs(small - gain * self._reference)
            level = max(self.min_contrast, 6.0)
        else:
            surface, spread = light_surface(small)
            diff = np.abs(small - surface)
            level = max(self.min_contrast, 5.0 * spread)
        mask = cv2.morphologyEx((diff > level).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        area, box = _largest(mask)
        present = area >= self.min_area
        h, w = small.shape
        self.box = (box[0] / w, box[1] / h, (box[0] + box[2]) / w, (box[1] + box[3]) / h) if present else None
        if self._reference is not None and not present and self._reference.shape == small.shape:
            step = REFERENCE_RATE * (small - self._reference)
            # Never far from the image a person chose, so a part left lying there cannot become "empty".
            self._reference = np.clip(self._reference + step, self._start - 15.0, self._start + 15.0)
        return present


_default = PresenceDetector()


def is_part_present(frame: np.ndarray, min_contrast: float = MIN_CONTRAST) -> bool:
    """Presence without a reference image."""
    detector = _default if min_contrast == MIN_CONTRAST else PresenceDetector(min_contrast=min_contrast)
    return detector(frame)


def learn_reference(frames: list[np.ndarray]) -> np.ndarray:
    """Median of frames of the empty station. Refuses frames that differ: something was moving."""
    if len(frames) < 5:
        raise ValueError(f"need at least 5 frames of the empty station, got {len(frames)}")
    median = np.median(np.stack(frames), axis=0).astype(np.uint8)
    moving = sum(PresenceDetector(reference=median)(f) for f in frames)
    if moving:
        raise ValueError(f"{moving} of {len(frames)} frames differ from the rest: the station was not empty or not still")
    return median


def main(argv=None) -> str:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p = parser.add_subparsers(dest="cmd", required=True).add_parser("learn")
    p.add_argument("--source", required=True, help="Camera index, RTSP address or video file")
    p.add_argument("--out", required=True)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    args = parser.parse_args(argv)
    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    if args.width and args.height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    frames = []
    while len(frames) < args.frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(args.out, learn_reference(frames)):
        parser.error(f"could not write {args.out}")
    print(f"Saved {args.out}. Put it in the camera's config as presence: {{reference: {args.out}}}")
    return args.out


if __name__ == "__main__":
    main()
