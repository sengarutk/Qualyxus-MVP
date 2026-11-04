"""Per-part inspection: turn a continuous frame stream into one verdict per physical part.

The frame-level detector fires on frames; the process model needs *parts*: every part
that passes the station must produce exactly one record - pass or fail - so good cycles
become negative labels too. A debounced presence state machine segments the stream into
part visits; the sharpest frame of each visit is kept for the anomaly detector.
"""
from __future__ import annotations

import collections
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

from src.telemetry.plc_adapter import PLC_FLAG_NOT_INSPECTED, PLC_PASS


@dataclass
class CompletedPart:
    camera_id: str
    first_ts: float
    last_ts: float
    best_frame: np.ndarray
    best_frame_id: int
    n_frames: int
    defect_event_ids: list[str] = field(default_factory=list)
    defect_types: list[str] = field(default_factory=list)

    @property
    def ts(self) -> float:
        """Representative inspection time: the middle of the visit."""
        return 0.5 * (self.first_ts + self.last_ts)


def part_pose(frame: np.ndarray) -> tuple[float, float, float] | None:
    """Part centre (x, y in pixels) and angle in degrees, or None if no part is found."""
    from src.inference.anomaly.preprocess import find_part

    rect = find_part(frame)
    if rect is None:
        return None
    (cx, cy), (w, h), angle = rect
    if w < h:  # the long side defines the angle, so it does not flip by 90 degrees
        angle += 90.0
    angle = (angle + 90.0) % 180.0 - 90.0
    return round(float(cx), 2), round(float(cy), 2), round(float(angle), 2)


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _verdict_of(flags: list[int], complete: bool) -> tuple[str, int]:
    rejects = [f for f in flags if f != PLC_PASS and f != PLC_FLAG_NOT_INSPECTED]
    if rejects:
        return "fail", max(rejects)
    if not complete or PLC_FLAG_NOT_INSPECTED in flags:
        return "not_inspected", PLC_FLAG_NOT_INSPECTED
    return "pass", PLC_PASS


class PartVerdictFuser:
    """One verdict per physical part, however many cameras judge it.

    A part passes only when every required view has passed it. A failure on any view fails
    the part. A required view that never reports within ``timeout_s`` makes the part
    not_inspected, so a dead camera can never let a part through as a pass. Without required
    views each camera's verdict is the part's verdict (a single-camera station).
    """

    def __init__(self, required_views: Iterable[str] = (), *, timeout_s: float = 5.0):
        self.required = frozenset(required_views)
        self.timeout_s = float(timeout_s)
        self._open: dict[str, dict[str, Any]] = {}
        self._closed: collections.deque[str] = collections.deque(maxlen=512)
        self._lock = threading.Lock()

    def add(self, cycle_id: str | None, camera_id: str, flag: int, now: float) -> dict[str, Any] | None:
        """Record one view's flag; returns the part's verdict once it is decided."""
        if not self.required or camera_id not in self.required:
            verdict, value = _verdict_of([flag], True)
            return {"cycle_id": cycle_id, "verdict": verdict, "flag": value, "views": {camera_id: flag}, "missing": []}
        if cycle_id is None:  # the other views of this part can never be matched to it
            verdict, value = _verdict_of([flag], False)
            return {"cycle_id": None, "verdict": verdict, "flag": value, "views": {camera_id: flag},
                    "missing": sorted(self.required - {camera_id})}
        with self._lock:
            if cycle_id in self._closed:
                return None  # a late view of a part that already timed out
            part = self._open.setdefault(cycle_id, {"deadline": now + self.timeout_s, "views": {}})
            part["views"][camera_id] = flag
            if self.required - set(part["views"]):
                return None
            return self._close(cycle_id)

    def expired(self, now: float) -> list[dict[str, Any]]:
        """Parts whose required views did not all report in time."""
        with self._lock:
            late = [c for c, part in self._open.items() if part["deadline"] <= now]
            return [self._close(c) for c in late]

    def _close(self, cycle_id: str) -> dict[str, Any]:
        part = self._open.pop(cycle_id)
        self._closed.append(cycle_id)
        missing = sorted(self.required - set(part["views"]))
        verdict, value = _verdict_of(list(part["views"].values()), not missing)
        return {"cycle_id": cycle_id, "verdict": verdict, "flag": value, "views": part["views"], "missing": missing}


class PartPresenceTracker:
    """Debounced part-visit segmentation for one camera."""

    def __init__(self, camera_id: str, *, on_frames: int = 2, off_frames: int = 2, max_visit_frames: int = 300):
        self.camera_id = camera_id
        self.on_frames = max(1, int(on_frames))
        self.off_frames = max(1, int(off_frames))
        self.max_visit_frames = max(1, int(max_visit_frames))
        self._present_run = 0
        self._absent_run = 0
        self._visit: dict | None = None
        self._await_removal = False   # a visit hit the length cap; the same part is still here

    def update(self, frame: np.ndarray, present: bool, ts: float, frame_id: int,
               defect_event_ids: list[str] | None = None, defect_types: list[str] | None = None) -> CompletedPart | None:
        if present:
            self._present_run += 1
            self._absent_run = 0
        else:
            self._absent_run += 1
            self._present_run = 0

        if self._await_removal:
            if self._absent_run < self.off_frames:
                return None
            self._await_removal = False

        if self._visit is None:
            if present and self._present_run >= self.on_frames:
                self._visit = {"first_ts": ts, "last_ts": ts, "best": None, "best_score": -1.0,
                               "best_id": frame_id, "n": 0, "events": [], "types": []}
            else:
                return None

        visit = self._visit
        if present:
            visit["last_ts"] = ts
            visit["n"] += 1
            score = sharpness(frame)
            if score > visit["best_score"]:
                visit["best_score"], visit["best"], visit["best_id"] = score, frame.copy(), frame_id
        visit["events"].extend(defect_event_ids or [])
        visit["types"].extend(defect_types or [])

        if not present and self._absent_run >= self.off_frames:
            return self._close()
        if visit["n"] >= self.max_visit_frames:
            self._await_removal = True
            return self._close()
        return None

    def _close(self) -> CompletedPart | None:
        visit, self._visit = self._visit, None
        if visit is None or visit["best"] is None:
            return None
        return CompletedPart(
            camera_id=self.camera_id,
            first_ts=visit["first_ts"],
            last_ts=visit["last_ts"],
            best_frame=visit["best"],
            best_frame_id=visit["best_id"],
            n_frames=visit["n"],
            defect_event_ids=list(dict.fromkeys(visit["events"])),
            defect_types=list(dict.fromkeys(visit["types"])),
        )
