from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.operations.action_tracker import ActionTracker
from src.telemetry.telemetry_provider import TelemetryProvider


@dataclass
class VerificationTask:
    action_id: str
    camera_id: str
    metric_name: str
    target_value: float
    tolerance: float = 0.1
    timeout: float = 15.0
    started_at: float = 0.0


class ActionVerifier:
    """Asynchronous actuator verification observer."""

    def __init__(
        self,
        telemetry_provider: TelemetryProvider,
        action_tracker: ActionTracker,
        poll_interval: float = 0.5,
    ):
        self.telemetry = telemetry_provider
        self.action_tracker = action_tracker
        self.poll_interval = float(poll_interval)
        self.active_verifications: dict[str, VerificationTask] = {}
        self._queue: queue.Queue[VerificationTask] = queue.Queue(maxsize=100)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.timeout_multiplier: float = 1.0

    def set_timeout_multiplier(self, multiplier: float) -> None:
        self.timeout_multiplier = max(1.0, float(multiplier))

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="action-verifier", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def watch_for_change(
        self,
        action_id: str,
        camera_id: str,
        metric_name: str,
        target_value: float,
        tolerance: float = 0.1,
        timeout: float = 15.0,
    ) -> bool:
        task = VerificationTask(
            action_id=action_id,
            camera_id=camera_id,
            metric_name=metric_name,
            target_value=float(target_value),
            tolerance=float(tolerance),
            timeout=float(timeout),
            started_at=time.time(),
        )
        try:
            self._queue.put_nowait(task)
            self.action_tracker.update_action(
                action_id,
                action_payload={
                    **(self.action_tracker.get_action(action_id) or {}).get("action_payload", {}),
                    "metric_name": metric_name,
                    "target_value": float(target_value),
                },
                changed_by="system:action_verifier",
                change_reason="verification_registered",
            )
            return True
        except queue.Full:
            return False

    def _mark_status(self, action_id: str, status: str) -> None:
        self.action_tracker.update_action(
            action_id,
            verification_status=status,
            verified_at=datetime.now(timezone.utc).isoformat() if status == "VERIFIED" else None,
            changed_by="system:action_verifier",
            change_reason=f"verification_{status.lower()}",
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                while True:
                    task = self._queue.get_nowait()
                    with self._lock:
                        self.active_verifications[task.action_id] = task
            except queue.Empty:
                pass

            now = time.time()
            to_remove: list[str] = []
            with self._lock:
                items = list(self.active_verifications.items())

            for action_id, task in items:
                try:
                    telemetry = self.telemetry.get_telemetry(task.camera_id)
                    current_value = getattr(telemetry, task.metric_name, None)
                    if current_value is not None and abs(float(current_value) - float(task.target_value)) <= float(task.tolerance):
                        self._mark_status(action_id, "VERIFIED")
                        to_remove.append(action_id)
                        continue
                    if now - task.started_at >= (task.timeout * self.timeout_multiplier):
                        self._mark_status(action_id, "FAILED")
                        to_remove.append(action_id)
                except Exception:
                    if now - task.started_at >= (task.timeout * self.timeout_multiplier):
                        self._mark_status(action_id, "TIMEOUT")
                        to_remove.append(action_id)

            if to_remove:
                with self._lock:
                    for action_id in to_remove:
                        self.active_verifications.pop(action_id, None)

            time.sleep(self.poll_interval)