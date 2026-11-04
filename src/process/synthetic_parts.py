"""Synthetic part imagery driven by the process simulator.

Renders an injection-molded part as a camera at an inspection station would see it,
with realistic nuisance variation (position jitter, small rotation, lighting changes)
and the visual signature of the defect the simulated process actually produced. Used to
train/validate the good-parts-only anomaly detector and to close the vision <-> process
loop end-to-end in simulation.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np

BACKGROUND = 40


def _part_geometry(width: int, height: int, rng: np.random.Generator) -> dict:
    cx = width / 2 + rng.uniform(-0.02, 0.02) * width
    cy = height / 2 + rng.uniform(-0.02, 0.02) * height
    return {
        "cx": cx,
        "cy": cy,
        "w": 0.56 * width,
        "h": 0.36 * height,
        "angle": rng.uniform(-2.0, 2.0),
        "gain": rng.uniform(0.95, 1.05),
    }


def render_part(width: int, height: int, defect_type: str | None, seed: int, *, return_mask: bool = False):
    """Render one part. With ``return_mask`` also returns the pixel mask of the defect."""
    rng = np.random.default_rng(seed)
    g = _part_geometry(width, height, rng)
    part = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
    dmask = np.zeros((height, width), dtype=np.uint8)
    x1, y1 = int(g["cx"] - g["w"] / 2), int(g["cy"] - g["h"] / 2)
    x2, y2 = int(g["cx"] + g["w"] / 2), int(g["cy"] + g["h"] / 2)
    radius = int(0.12 * g["h"])
    body = (150, 150, 156)

    # Rounded rectangle body
    cv2.rectangle(part, (x1 + radius, y1), (x2 - radius, y2), body, -1)
    cv2.rectangle(part, (x1, y1 + radius), (x2, y2 - radius), body, -1)
    for cx_, cy_ in ((x1 + radius, y1 + radius), (x2 - radius, y1 + radius), (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)):
        cv2.circle(part, (cx_, cy_), radius, body, -1)
    # Mounting hole and stiffening ribs
    hole_c = (int(x1 + 0.2 * g["w"]), int(g["cy"]))
    cv2.circle(part, hole_c, int(0.18 * g["h"]), (BACKGROUND,) * 3, -1)
    cv2.circle(part, hole_c, int(0.18 * g["h"]), (120, 120, 126), 2)
    for frac in (0.45, 0.62, 0.79):
        xr = int(x1 + frac * g["w"])
        cv2.line(part, (xr, y1 + int(0.15 * g["h"])), (xr, y2 - int(0.15 * g["h"])), (175, 175, 182), 3)

    if defect_type == "short_shot":
        # Incomplete fill at the far end of the flow path: ragged missing chunk.
        depth = rng.uniform(0.14, 0.24) * g["w"]
        xs = x2 - depth
        pts = [(x2 + 5, y1 - 5), (int(xs), y1 - 5)]
        for k in range(1, 7):
            pts.append((int(xs + rng.uniform(-0.05, 0.05) * g["w"]), int(y1 + k * (y2 - y1) / 7)))
        pts += [(int(xs), y2 + 5), (x2 + 5, y2 + 5)]
        body_mask = (part[..., 0] > BACKGROUND + 20).astype(np.uint8)  # material that should be there
        cv2.fillPoly(dmask, [np.array(pts, dtype=np.int32)], 1)
        dmask &= body_mask
        cv2.fillPoly(part, [np.array(pts, dtype=np.int32)], (BACKGROUND,) * 3)
    elif defect_type == "flash":
        # Thin film of excess material escaping at the parting line.
        start = int(x1 + rng.uniform(0.3, 0.5) * g["w"])
        length = int(rng.uniform(0.25, 0.4) * g["w"])
        top = rng.random() < 0.5
        y = y1 if top else y2
        thickness = int(rng.integers(5, 9))
        y_a, y_b = (y - thickness, y) if top else (y, y + thickness)
        cv2.rectangle(part, (start, y_a), (start + length, y_b), (198, 198, 205), -1)
        cv2.rectangle(dmask, (start, y_a), (start + length, y_b), 1, -1)
    elif defect_type == "sink_mark":
        # Local depression over a thick section: a soft dark blemish.
        layer = np.zeros(part.shape[:2], dtype=np.float32)
        cx_ = int(x1 + rng.uniform(0.45, 0.8) * g["w"])
        cy_ = int(g["cy"] + rng.uniform(-0.2, 0.2) * g["h"])
        axes = (int(rng.uniform(0.08, 0.12) * g["w"]), int(rng.uniform(0.12, 0.2) * g["h"]))
        cv2.ellipse(layer, (cx_, cy_), axes, rng.uniform(0, 180), 0, 360, 1.0, -1)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=6)
        part = np.clip(part.astype(np.float32) - 60.0 * layer[..., None], 0, 255).astype(np.uint8)
        dmask = (layer > 0.25).astype(np.uint8)
    elif defect_type == "burn_mark":
        # Dieseling at the end of fill: dark brown streaks.
        base_x = int(x2 - rng.uniform(0.08, 0.18) * g["w"])
        for _ in range(int(rng.integers(2, 4))):
            pts = np.array([
                (base_x + int(rng.uniform(-10, 10)), int(y1 + rng.uniform(0.1, 0.3) * g["h"])),
                (base_x + int(rng.uniform(-14, 14)), int(g["cy"])),
                (base_x + int(rng.uniform(-10, 10)), int(y2 - rng.uniform(0.1, 0.3) * g["h"])),
            ], dtype=np.int32)
            thickness = int(rng.integers(4, 7))
            cv2.polylines(part, [pts], False, (25, 40, 70), thickness)
            cv2.polylines(dmask, [pts], False, 1, thickness)

    # Nuisance variation: rotation, lighting and sensor noise.
    rot = cv2.getRotationMatrix2D((g["cx"], g["cy"]), g["angle"], 1.0)
    part = cv2.warpAffine(part, rot, (width, height), borderValue=(BACKGROUND,) * 3)
    part = np.clip(part.astype(np.float32) * g["gain"] + rng.normal(0.0, 2.0, part.shape), 0, 255)
    if return_mask:
        dmask = cv2.warpAffine(dmask, rot, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
        return part.astype(np.uint8), dmask
    return part.astype(np.uint8)


@dataclass
class _ScheduledPart:
    cycle_id: str
    defect_type: str | None
    appear_ts: float
    leave_ts: float
    seed: int


class SyntheticPartFeed:
    """Schedules simulated parts in front of a synthetic camera.

    A part produced by cycle *c* arrives at the inspection station ``transport_delay_s``
    after the cycle ends and dwells there for ``dwell_s``.

    With ``n_views > 1`` this feed is one face of a multi-camera station: every view sees
    the same part, but a defect sits on a single face, so only one view can catch it.
    ``inject_defect_every`` places a demo defect on every Nth part regardless of the process.
    """

    def __init__(self, source, *, transport_delay_s: float = 4.0, dwell_s: float = 0.6,
                 view_index: int = 0, n_views: int = 1, inject_defect_every: int = 0):
        self.source = source
        self.transport_delay_s = float(transport_delay_s)
        self.dwell_s = float(dwell_s)
        self.view_index = int(view_index)
        self.n_views = max(1, int(n_views))
        self.inject_defect_every = int(inject_defect_every)
        self._scheduled: dict[str, _ScheduledPart] = {}
        self._lock = threading.Lock()

    def part_in_view(self, now: float) -> _ScheduledPart | None:
        window_start = now - self.transport_delay_s - self.dwell_s - 30.0
        for cycle, outcome in self.source.outcomes_ended_between(window_start, now - self.transport_delay_s + 1e-6):
            appear = cycle.end_ts + self.transport_delay_s
            if appear <= now < appear + self.dwell_s:
                with self._lock:
                    part = self._scheduled.get(cycle.cycle_id)
                    if part is None:
                        part = _ScheduledPart(cycle.cycle_id, self._defect_for(cycle.sequence, outcome.defect_type),
                                              appear, appear + self.dwell_s, seed=int(cycle.sequence))
                        self._scheduled[cycle.cycle_id] = part
                        if len(self._scheduled) > 256:
                            self._scheduled.pop(next(iter(self._scheduled)))
                return part
        return None

    def _defect_for(self, sequence: int, process_defect: str | None) -> str | None:
        from src.process.simulator import DEFECT_TYPES

        every = self.inject_defect_every
        if every > 0 and sequence % every == 0:
            return DEFECT_TYPES[(sequence // every) % len(DEFECT_TYPES)]
        return process_defect

    def render(self, now: float, width: int, height: int) -> tuple[np.ndarray, _ScheduledPart | None]:
        part = self.part_in_view(now)
        if part is None:
            frame = np.full((height, width, 3), BACKGROUND, dtype=np.uint8)
            noise = np.random.default_rng(int(now * 1000) & 0xFFFFFFFF).normal(0.0, 0.6, frame.shape)
            return np.clip(frame + noise, 0, 255).astype(np.uint8), None
        defect = part.defect_type if part.seed % self.n_views == self.view_index else None
        return render_part(width, height, defect, part.seed + self.view_index * 7919), part


def export_segmentation_dataset(out_dir, *, n_train: int = 400, n_val: int = 100, width: int = 640,
                                height: int = 480, seed: int = 0, good_fraction: float = 0.2) -> str:
    """Write a labelled YOLO-seg dataset (data.yaml + images/labels) of rendered parts.

    Classes are the four simulated defect types; a share of good parts is included as
    background-only images (empty label files), as in a real defect dataset.
    """
    from pathlib import Path

    from src.process.simulator import DEFECT_TYPES

    out = Path(out_dir)
    rng = np.random.default_rng(seed)
    for split, n in (("train", n_train), ("val", n_val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            defect = None if rng.random() < good_fraction else DEFECT_TYPES[int(rng.integers(len(DEFECT_TYPES)))]
            img, mask = render_part(width, height, defect, seed=int(rng.integers(1 << 31)), return_mask=True)
            stem = f"{split}_{i:05d}"
            cv2.imwrite(str(out / "images" / split / f"{stem}.png"), img)
            lines = []
            if defect is not None:
                cls = DEFECT_TYPES.index(defect)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    if cv2.contourArea(c) < 8 or len(c) < 3:
                        continue
                    coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in c.reshape(-1, 2))
                    lines.append(f"{cls} {coords}")
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(DEFECT_TYPES))
    (out / "data.yaml").write_text(
        f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/val\nnames:\n{names}\n"
    )
    return str(out / "data.yaml")
