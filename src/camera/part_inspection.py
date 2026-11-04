"""Per-part inspection: turn a continuous frame stream into one verdict per physical part.

The frame-level detector fires on frames; the process model needs *parts*: every part
that passes the station must produce exactly one record - pass or fail - so good cycles
become negative labels too. A debounced presence state machine segments the stream into
part visits; the sharpest frame of each visit is kept for the anomaly detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


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


def sharpness(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


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

    def update(self, frame: np.ndarray, present: bool, ts: float, frame_id: int,
               defect_event_ids: list[str] | None = None, defect_types: list[str] | None = None) -> CompletedPart | None:
        if present:
            self._present_run += 1
            self._absent_run = 0
        else:
            self._absent_run += 1
            self._present_run = 0

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

        if (not present and self._absent_run >= self.off_frames) or visit["n"] >= self.max_visit_frames:
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
