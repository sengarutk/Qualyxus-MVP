"""Part trigger from the PLC, for stations where the picture alone cannot say a part has arrived.

The PLC adds 1 to the ``trigger`` register each time a part is in position. Every camera then
keeps the sharpest of its next few frames: no image analysis decides whether there is a part,
and all views of one trigger are known to be the same part. A trigger a camera has no image for
is reported as not inspected, and so is a trigger that was missed because the next came too soon.

Polling takes tens of milliseconds, so the part must be at rest (or the trigger raised early).
A part that does not stop needs a camera wired to the sensor.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from src.camera.part_inspection import CompletedPart
from src.inference.anomaly.image_quality import sharpness
from src.telemetry.plc_adapter import PLCAdapter, PLCAdapterConfig

logger = logging.getLogger("QualyxusTrigger")

Missed = list[tuple[str, str, float]]       # (camera, part key, trigger time)


class PartTrigger:
    """Collects each camera's frames after a trigger."""

    def __init__(self, cameras: list[str], *, frames: int = 3, delay_s: float = 0.05, timeout_s: float = 5.0):
        self.cameras = list(cameras)
        self.frames, self.delay_s, self.timeout_s = max(1, int(frames)), float(delay_s), float(timeout_s)
        self._open: dict[str, dict] = {}
        self._lock = threading.Lock()

    def fire(self, key: str, ts: float) -> Missed:
        """Start a part on every camera; returns the views the previous part never got."""
        missed = []
        with self._lock:
            for camera in self.cameras:
                old = self._open.get(camera)
                if old is not None:
                    missed.append((camera, old["key"], old["ts"]))
                self._open[camera] = {"key": key, "ts": ts, "n": 0, "best": None, "score": -1.0, "id": 0,
                                      "first": ts, "last": ts, "events": [], "types": []}
        return missed

    def frame(self, camera: str, frame: np.ndarray, frame_id: int, ts: float, events: list[str],
              types: list[str]) -> CompletedPart | None:
        with self._lock:
            part = self._open.get(camera)
            if part is None or ts < part["ts"] + self.delay_s:
                return None
            score = sharpness(frame)
            if score > part["score"]:
                part.update(best=frame.copy(), score=score, id=frame_id)
            part["n"] += 1
            part["last"] = ts
            part["events"] += events
            part["types"] += types
            if part["n"] < self.frames:
                return None
            del self._open[camera]
        # The trigger's time, the same for every camera, is the part's time.
        return CompletedPart(camera_id=camera, first_ts=part["ts"], last_ts=part["ts"], best_frame=part["best"],
                             best_frame_id=part["id"], n_frames=part["n"],
                             defect_event_ids=list(dict.fromkeys(part["events"])),
                             defect_types=list(dict.fromkeys(part["types"])), trigger_key=part["key"])

    def expire(self, now: float) -> Missed:
        with self._lock:
            late = [(c, p["key"], p["ts"]) for c, p in self._open.items() if now - p["ts"] > self.timeout_s]
            for camera, _, _ in late:
                del self._open[camera]
        return late


class TriggerSource:
    """Polls the PLC's trigger counter on its own connection, so verdict writes never delay it."""

    def __init__(self, config: PLCAdapterConfig, on_trigger: Callable[[int, int], None], *, poll_s: float = 0.02):
        self.adapter = PLCAdapter(config)
        self.on_trigger, self.poll_s = on_trigger, float(poll_s)
        self._last: int | None = None
        self._read_at = float("-inf")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._read_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="part-trigger", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.adapter.close()

    def alive(self, max_age_s: float = 2.0) -> bool:
        return time.monotonic() - self._read_at < max_age_s

    def poll(self) -> bool:
        """False when the PLC could not be read."""
        value = self.adapter._read("trigger")
        if value is None:
            return False
        self._read_at = time.monotonic()
        last, self._last = self._last, value
        if last is None or value == last:
            return True
        step = (value - last) & 0xFFFF
        if step > 1000:         # a counter reset, not a thousand parts
            logger.warning("[Trigger] Counter went from %d to %d: taken as a reset.", last, value)
            step = 1
        if step > 1:
            logger.error("[Trigger] %d triggers came faster than they could be read.", step - 1)
        self.on_trigger(value, step - 1)
        return True

    def _run(self) -> None:
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                readable = self.poll()
            except Exception as e:
                logger.error("[Trigger] %s", e, exc_info=True)
                readable = True
            # An unreachable PLC is asked twice a second, not fifty times.
            wait = self.poll_s if readable else 0.5
            self._stop.wait(max(0.0, wait - (time.monotonic() - started)))
