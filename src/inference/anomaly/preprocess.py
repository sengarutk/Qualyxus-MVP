"""Part registration and tiling in front of any anomaly detector.

Registration
    Anomaly models compare every patch with what that region normally looks like, so
    a part that lands shifted or rotated in the frame looks "abnormal" everywhere. The
    registrar finds the part against the background (border-colour model + Otsu), fits a
    minimum-area rectangle, rotates it to a canonical orientation, crops it with a
    margin and resizes it to a canonical size learned from the training parts. The
    remaining 180-degree ambiguity is resolved by correlating with a reference template
    of the registered training parts. Anomaly maps are warped back onto the original
    frame, so overlays and boxes stay in camera coordinates.

Tiling
    Small defects on large parts vanish when the whole image is resized to the model's
    input (e.g. a 0.3 mm flash on a 200 mm part). With ``tiles=(rows, cols)`` the
    (registered) image is cut into overlapping tiles, every tile is scored at full model
    resolution, the maps are stitched (max over overlaps) and the image score is the
    maximum tile score. Training uses all tiles of all good images.

The whole state is JSON-serialisable, so it travels with PyTorch checkpoints and with
ONNX sidecars alike.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger("QualyxusPreprocess")

_TEMPLATE_SIZE = 64


@dataclass
class View:
    """One image handed to the detector, and how its map goes back to the frame."""
    image: np.ndarray
    box: tuple[int, int, int, int]   # x1, y1, x2, y2 inside the canonical image


@dataclass
class Prepared:
    views: list[View]
    canonical_shape: tuple[int, int]          # (h, w) of the canonical image
    warp: np.ndarray | None                   # 2x3 affine frame -> canonical (None = identity)
    frame_shape: tuple[int, int]              # (h, w) of the original frame
    registered: bool = True


def _border_background(img: np.ndarray, width: int = 6) -> np.ndarray:
    border = np.concatenate([img[:width].reshape(-1, 3), img[-width:].reshape(-1, 3),
                             img[:, :width].reshape(-1, 3), img[:, -width:].reshape(-1, 3)])
    return np.median(border, axis=0)


def find_part(img: np.ndarray, min_area_fraction: float = 0.01) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Minimum-area rectangle ((cx, cy), (w, h), angle) of the dominant foreground object."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    bg = _border_background(img.astype(np.float32))
    diff = np.abs(img.astype(np.float32) - bg).max(axis=2)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    diff8 = np.clip(diff, 0, 255).astype(np.uint8)
    level, mask = cv2.threshold(diff8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if level < 8:  # no real contrast: nothing distinct from the background
        return None
    k = max(3, int(round(min(img.shape[:2]) * 0.01)) | 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area_fraction * img.shape[0] * img.shape[1]:
        return None
    return cv2.minAreaRect(contour)


def _canonical_rect(rect) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Same rectangle expressed with the long side horizontal (angle in (-90, 90])."""
    (cx, cy), (w, h), angle = rect
    if h > w:
        w, h, angle = h, w, angle + 90.0
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return (cx, cy), (w, h), angle


def _warp_for(rect, out_w: int, out_h: int, margin: float, flip: bool) -> np.ndarray:
    (cx, cy), (w, h), angle = rect
    scale = min(out_w / (w * (1 + 2 * margin)), out_h / (h * (1 + 2 * margin)))
    m = cv2.getRotationMatrix2D((cx, cy), angle + (180.0 if flip else 0.0), scale)
    m[0, 2] += out_w / 2.0 - cx
    m[1, 2] += out_h / 2.0 - cy
    return m


def _template(img: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (_TEMPLATE_SIZE, _TEMPLATE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-6)


def tile_boxes(h: int, w: int, rows: int, cols: int, overlap: float) -> list[tuple[int, int, int, int]]:
    th = int(np.ceil(h / (rows - (rows - 1) * overlap)))
    tw = int(np.ceil(w / (cols - (cols - 1) * overlap)))
    ys = [0] if rows == 1 else np.linspace(0, h - th, rows).round().astype(int).tolist()
    xs = [0] if cols == 1 else np.linspace(0, w - tw, cols).round().astype(int).tolist()
    return [(x, y, min(w, x + tw), min(h, y + th)) for y in ys for x in xs]


@dataclass
class Preprocessor:
    register: bool = False
    tiles: tuple[int, int] | None = None
    overlap: float = 0.15
    margin: float = 0.06
    canonical_size: tuple[int, int] | None = None     # (w, h); learned at fit when None
    template: np.ndarray | None = field(default=None, repr=False)
    fit_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.register or (self.tiles is not None and tuple(self.tiles) != (1, 1))

    # ------------------------------------------------------------------ fit

    def fit(self, images: Sequence[np.ndarray]) -> dict[str, Any]:
        if not self.register:
            return {}
        rects = [find_part(img) for img in images]
        found = [_canonical_rect(r) for r in rects if r is not None]
        if len(found) < max(3, int(0.8 * len(images))):
            raise ValueError(f"Part registration found the part in only {len(found)}/{len(images)} good images: "
                             "check lighting/background contrast or train without registration")
        if self.canonical_size is None:
            w_med = float(np.median([r[1][0] for r in found]))
            h_med = float(np.median([r[1][1] for r in found]))
            # Keep the part's aspect ratio; long side ~ the median part length, multiple of 16.
            scale = min(1.0, 1024.0 / (w_med * (1 + 2 * self.margin)))
            out_w = int(round(w_med * (1 + 2 * self.margin) * scale / 16)) * 16
            out_h = int(round(h_med * (1 + 2 * self.margin) * scale / 16)) * 16
            self.canonical_size = (max(32, out_w), max(32, out_h))
        # Reference template: average of the registered parts in their first-found orientation,
        # each aligned (0 or 180 degrees) to the running reference.
        ref = None
        for img, rect in zip(images, rects):
            if rect is None:
                continue
            warped = self._apply(img, _canonical_rect(rect), flip=False)
            t = _template(warped)
            if ref is not None and float((t * ref).mean()) < float((t[::-1, ::-1] * ref).mean()):
                t = t[::-1, ::-1]
            ref = t if ref is None else ref + t
        self.template = (ref - ref.mean()) / (ref.std() + 1e-6)
        self.fit_stats = {"registered": len(found), "images": len(images), "canonical_size": list(self.canonical_size)}
        logger.info("[Registration] canonical part %sx%s px from %d/%d good images",
                    *self.canonical_size, len(found), len(images))
        return dict(self.fit_stats)

    # ------------------------------------------------------------ transform

    def _apply(self, img: np.ndarray, rect, flip: bool) -> np.ndarray:
        out_w, out_h = self.canonical_size
        m = _warp_for(rect, out_w, out_h, self.margin, flip)
        return cv2.warpAffine(img, m, (out_w, out_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def prepare(self, img: np.ndarray) -> Prepared:
        frame_shape = img.shape[:2]
        warp, canonical, registered = None, img, True
        if self.register:
            rect = find_part(img)
            if rect is None:
                registered = False   # nothing found: score the frame as-is (it will look abnormal)
                canonical = cv2.resize(img, tuple(self.canonical_size), interpolation=cv2.INTER_AREA)
                sx, sy = self.canonical_size[0] / img.shape[1], self.canonical_size[1] / img.shape[0]
                warp = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float64)
            else:
                rect = _canonical_rect(rect)
                out_w, out_h = self.canonical_size
                flip = False
                if self.template is not None:
                    t = _template(self._apply(img, rect, flip=False))
                    flip = float((t[::-1, ::-1] * self.template).mean()) > float((t * self.template).mean())
                warp = _warp_for(rect, out_w, out_h, self.margin, flip)
                canonical = cv2.warpAffine(img, warp, (out_w, out_h), flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE)
        h, w = canonical.shape[:2]
        if self.tiles and tuple(self.tiles) != (1, 1):
            boxes = tile_boxes(h, w, int(self.tiles[0]), int(self.tiles[1]), float(self.overlap))
        else:
            boxes = [(0, 0, w, h)]
        views = [View(canonical[y1:y2, x1:x2], (x1, y1, x2, y2)) for x1, y1, x2, y2 in boxes]
        return Prepared(views, (h, w), warp, frame_shape, registered)

    def views(self, img: np.ndarray) -> list[np.ndarray]:
        return [v.image for v in self.prepare(img).views]

    def combine(self, prepared: Prepared, results: Sequence[tuple[float, np.ndarray]]) -> tuple[float, np.ndarray]:
        """Stitch per-view (score, map) results into one image score and a frame-sized map."""
        h, w = prepared.canonical_shape
        canvas = np.zeros((h, w), dtype=np.float32)
        for view, (_, amap) in zip(prepared.views, results):
            x1, y1, x2, y2 = view.box
            if amap.shape != (y2 - y1, x2 - x1):
                amap = cv2.resize(amap, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
            np.maximum(canvas[y1:y2, x1:x2], amap, out=canvas[y1:y2, x1:x2])
        score = float(max(s for s, _ in results))
        if prepared.warp is not None:
            fh, fw = prepared.frame_shape
            canvas = cv2.warpAffine(canvas, prepared.warp, (fw, fh), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        return score, canvas

    # -------------------------------------------------------------- persist

    def state(self) -> dict[str, Any]:
        return {
            "register": self.register,
            "tiles": list(self.tiles) if self.tiles else None,
            "overlap": self.overlap,
            "margin": self.margin,
            "canonical_size": list(self.canonical_size) if self.canonical_size else None,
            "template": self.template.round(4).tolist() if self.template is not None else None,
            "fit_stats": self.fit_stats,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any] | None) -> "Preprocessor":
        state = state or {}
        template = state.get("template")
        return cls(
            register=bool(state.get("register", False)),
            tiles=tuple(state["tiles"]) if state.get("tiles") else None,
            overlap=float(state.get("overlap", 0.15)),
            margin=float(state.get("margin", 0.06)),
            canonical_size=tuple(state["canonical_size"]) if state.get("canonical_size") else None,
            template=np.asarray(template, dtype=np.float32) if template is not None else None,
            fit_stats=dict(state.get("fit_stats") or {}),
        )
