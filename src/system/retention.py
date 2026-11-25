"""Keep the databases a bounded size: old rows go, the audit trail is archived by a person.

``process.retention_days`` (cycles, inspections), ``logging.retention_days`` (defect events) and
``logging.tracking_retention_days``; 0 keeps everything. The audit trail is never deleted here:
move its old days out with ``python -m src.inference.audit_log archive``.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger("QualyxusRetention")

DAY = 86400.0


class RetentionWorker:
    def __init__(self, cfg: dict[str, Any], *, store=None, event_logger=None, every_s: float = 6 * 3600.0):
        process, events = cfg.get("process") or {}, cfg.get("logging") or {}
        self.process_days = float(process.get("retention_days", 730))
        self.event_days = float(events.get("retention_days", 730))
        self.tracking_days = float(events.get("tracking_retention_days", 30))
        self.store, self.event_logger, self.every_s = store, event_logger, float(every_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def run_once(self) -> dict[str, int]:
        done: dict[str, int] = {}
        for name, job in (("process", self._process), ("events", self._events)):
            try:
                done.update(job())
            except Exception as e:      # tried again at the next round; nothing is lost by waiting
                logger.error("[Retention] %s not pruned: %s", name, e)
        if any(done.values()):
            logger.info("[Retention] Removed %s", {k: v for k, v in done.items() if v})
        return done

    def _process(self) -> dict[str, int]:
        return self.store.prune(self.process_days * DAY) if self.store is not None and self.process_days > 0 else {}

    def _events(self) -> dict[str, int]:
        if self.event_logger is None:
            return {}
        forever = 1e4 * DAY
        return self.event_logger.prune(events_older_than_s=self.event_days * DAY if self.event_days > 0 else forever,
                                       tracking_older_than_s=self.tracking_days * DAY if self.tracking_days > 0
                                       else forever)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="retention", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        wait = 60.0
        while not self._stop.wait(wait):
            self.run_once()
            wait = self.every_s
