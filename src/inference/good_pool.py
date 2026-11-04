"""Rolling pool of recently passed parts, per recipe: the training set for retraining.

When imaging conditions change (new lighting, camera moved, new material lot) the
anomaly model has to be retrained on good parts *as they look now*. The pool keeps a
sampled, bounded, timestamped set of passed parts so an approved retraining request
can train on parts captured after the change point without anyone collecting images.

Only parts every model passed are pooled, and parts in the review band (borderline
scores) are held back until a human confirms them, so the pool cannot silently absorb
defects. Retraining itself always requires a human approval (see model_registry).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import cv2
import numpy as np


class GoodPartPool:
    def __init__(self, directory: str | Path, recipe: str, *, every_n: int = 5, max_items: int = 3000,
                 max_score: float | None = 0.8):
        self.dir = Path(directory) / recipe
        self.every_n = max(1, int(every_n))
        self.max_items = int(max_items)
        self.max_score = max_score
        self._seen = 0
        self._lock = threading.Lock()

    def offer(self, image: np.ndarray, *, verdict: str, normalized_score: float | None, camera_id: str,
              image_sha: str | None = None) -> Path | None:
        """Pool every n-th confidently-good part. Returns the saved path, if any."""
        if verdict != "pass":
            return None
        if self.max_score is not None and normalized_score is not None and normalized_score > self.max_score:
            return None
        with self._lock:
            self._seen += 1
            if self._seen % self.every_n:
                return None
            self.dir.mkdir(parents=True, exist_ok=True)
            ts = time.time()
            stem = f"{int(ts * 1000):013d}_{uuid.uuid4().hex[:6]}_{camera_id}"  # sortable, collision-free
            path = self.dir / f"{stem}.png"
            cv2.imwrite(str(path), image)
            (self.dir / f"{stem}.json").write_text(json.dumps(
                {"ts": ts, "camera_id": camera_id, "normalized_score": normalized_score, "image_sha": image_sha}),
                encoding="utf-8")
            self._evict()
            return path

    def _evict(self) -> None:
        images = sorted(self.dir.glob("*.png"))
        for old in images[: max(0, len(images) - self.max_items)]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)

    def entries(self, since_ts: float | None = None) -> list[Path]:
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.glob("*.png")):
            try:
                ts = int(p.stem.split("_", 1)[0]) / 1000.0
            except ValueError:
                continue
            if since_ts is None or ts >= since_ts:
                out.append(p)
        return out

    def load(self, *, since_ts: float | None = None, limit: int | None = None) -> list[np.ndarray]:
        paths = self.entries(since_ts)
        if limit:
            paths = paths[-int(limit):]
        images = [cv2.imread(str(p)) for p in paths]
        return [img for img in images if img is not None]

    def __len__(self) -> int:
        return len(self.entries())
