"""Rolling pool of recently passed parts, per recipe: the training set for retraining.

When imaging conditions change (new lighting, camera moved, new material lot) the
anomaly model has to be retrained on good parts *as they look now*. The pool keeps a
sampled, bounded, timestamped set of passed parts so an approved retraining request
can train on parts captured after the change point without anyone collecting images.

Only parts every model passed are pooled, and parts in the review band (borderline
scores) are held back until a human confirms them, so the pool cannot silently absorb
defects. That makes the pool easy parts only, so it must not set a decision threshold:
calibration uses a second, unbiased sample (``<recipe>/calibration``) of every n-th part
whatever it scored. Those the model is sure about count as good; the rest go to the review
queue and count only once a person has confirmed them. Retraining itself always requires a human approval (see model_registry).
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
                 max_score: float | None = 0.8, calibration_every_n: int = 20, calibration_max_items: int = 2000):
        self.dir = Path(directory) / recipe
        self.every_n = max(1, int(every_n))
        self.max_items = int(max_items)
        self.max_score = max_score
        self.calibration_dir = self.dir / "calibration"
        self.calibration_every_n = max(1, int(calibration_every_n))
        self.calibration_max_items = int(calibration_max_items)
        self._seen = 0
        self._parts = 0
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
            path = self._write(self.dir, image, camera_id, normalized_score=normalized_score, image_sha=image_sha)
            self._evict(self.dir, self.max_items)
            return path

    def sample(self, image: np.ndarray, *, verdict: str, normalized_score: float | None, camera_id: str,
               image_sha: str | None) -> str | None:
        """Every n-th judged part for calibration: "accepted", "needs_review" or None (not sampled)."""
        with self._lock:
            self._parts += 1
            if self._parts % self.calibration_every_n:
                return None
            accepted = verdict == "pass" and (self.max_score is None or normalized_score is None
                                              or normalized_score <= self.max_score)
            self._write(self.calibration_dir, image, camera_id, normalized_score=normalized_score,
                        image_sha=image_sha, accepted=accepted)
            self._evict(self.calibration_dir, self.calibration_max_items)
            return "accepted" if accepted else "needs_review"

    def calibration_set(self, labels: dict[str, str], *, since_ts: float | None = None
                        ) -> tuple[list[np.ndarray], int]:
        """(good images, parts still waiting for a person). ``labels``: image_sha -> "pass" / "fail".

        Leaving out the waiting parts would leave out exactly the hard ones, so the caller must not
        calibrate until there are none.
        """
        images, waiting = [], 0
        for path in self._list(self.calibration_dir, since_ts):
            try:
                meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                waiting += 1
                continue
            label = labels.get(meta.get("image_sha")) or ("pass" if meta.get("accepted") else None)
            if label is None:
                waiting += 1
            elif label == "pass":
                image = cv2.imread(str(path))
                if image is not None:
                    images.append(image)
        return images, waiting

    @staticmethod
    def _write(directory: Path, image: np.ndarray, camera_id: str, **meta) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        ts = time.time()
        path = directory / f"{int(ts * 1000):013d}_{uuid.uuid4().hex[:6]}_{camera_id}.png"  # sortable, collision-free
        sidecar = path.with_suffix(".json")
        try:
            sidecar.write_text(json.dumps({"ts": ts, "camera_id": camera_id, **meta}), encoding="utf-8")
            if not cv2.imwrite(str(path), image):
                raise OSError(f"could not write {path}")
        except OSError:
            path.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
            raise
        return path

    @staticmethod
    def _evict(directory: Path, max_items: int) -> None:
        images = sorted(directory.glob("*.png"))
        for old in images[: max(0, len(images) - max_items)]:
            old.unlink(missing_ok=True)
            old.with_suffix(".json").unlink(missing_ok=True)

    def entries(self, since_ts: float | None = None) -> list[Path]:
        return self._list(self.dir, since_ts)

    @staticmethod
    def _list(directory: Path, since_ts: float | None = None) -> list[Path]:
        if not directory.exists():
            return []
        out = []
        for p in sorted(directory.glob("*.png")):
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
