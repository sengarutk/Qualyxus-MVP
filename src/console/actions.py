"""What a signed-in person can change from the console.

Every change is written to the audit trail first (who, in which role, what, why). If that
cannot be written, the change does not happen.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from src.inference.audit_log import record_person
from src.inference.labeling import prioritise, review_items
from src.inference.model_registry import VisionModelRegistry
from src.operations.action_tracker import ActionTracker
from src.process.store import MAINTENANCE, PROCESS_CHANGE

Person = tuple[str, str]
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MIN_REASON = 3


class ActionError(ValueError):
    """The change is not possible; the text is shown to the person."""


class ConsoleActions:
    def __init__(self, cfg: dict[str, Any], data):
        self.cfg, self.data = cfg, data
        inspection = cfg.get("inspection") or {}
        self.review_dir = Path((inspection.get("review") or {}).get("dir", "data/review"))
        self.registry_dir = inspection.get("registry_dir")
        self.audit = inspection.get("audit") or {}

    # ------------------------------------------------------------------ read

    def work(self) -> dict[str, Any]:
        """Everything waiting for a person."""
        actions = []
        if Path(self.data.actions_db).exists():
            tracker = ActionTracker(sqlite_path=self.data.actions_db)
            for status in ("open", "in_progress"):
                for a in tracker.list_actions(status=status, limit=100):
                    payload = a.get("action_payload") or {}
                    actions.append({"id": a["action_id"], "status": a["status"], "type": payload.get("action_type"),
                                    "title": payload.get("title"), "text": payload.get("explanation"),
                                    "issued": a.get("timestamp"), "needs_approval": bool(a.get("approval_required")),
                                    "approved_by": a.get("approved_by"), "assigned_to": a.get("assigned_to")})
        requests, models = [], []
        if self.registry_dir and Path(self.registry_dir).exists():
            registry = VisionModelRegistry(self.registry_dir)
            requests = [{k: r.get(k) for k in ("id", "recipe", "camera_id", "reason", "status", "decided_by", "created_ts")}
                        for r in registry.requests() if r["status"] in ("pending", "approved")]
            for row in registry.overview():
                if row["role"] == "anomaly":
                    report = registry.shadow_report(row["recipe"], review_dir=self.review_dir) if row["shadow"] else None
                    models.append({**{k: row[k] for k in ("recipe", "active", "shadow", "versions", "last_change")},
                                   "shadow_report": report})
        waiting = prioritise(review_items(self.review_dir)) if self.review_dir.exists() else []
        review = [{"id": i["id"], "camera": i.get("camera_id"), "ts": i.get("ts"), "score": i.get("normalized_score"),
                   "verdict": i.get("model_verdict"), "cycle_id": i.get("cycle_id")} for i in waiting[:24]]
        return {"actions": actions, "requests": requests, "models": models, "review": review,
                "review_waiting": len(waiting), "machines": self.data.machines}

    def _review_path(self, camera: str, item: str) -> Path | None:
        if not _NAME.match(camera) or not _NAME.match(item):
            return None
        path = self.review_dir / camera / f"{item}.json"
        return path if path.is_file() else None

    def review_image(self, camera: str, item: str, overlay: bool) -> Path | None:
        meta = self._review_path(camera, item)
        if meta is None:
            return None
        names = [f"{item}_overlay.jpg", f"{item}_overlay.png"] if overlay else [f"{item}.png"]
        return next((meta.with_name(n) for n in names if meta.with_name(n).is_file()), None)

    # ---------------------------------------------------------------- change

    def _record(self, person: Person, what: str, target: str, reason: str, **details) -> None:
        if not self.audit.get("enabled", True):
            return
        try:
            record_person(self.audit.get("dir", "data/audit"), self.audit.get("key_file"), by=person[0], role=person[1],
                          what=what, target=target, reason=reason, **details)
        except Exception as e:
            raise ActionError(f"Not done: it could not be written to the audit trail ({e})") from e

    @staticmethod
    def _reason(reason: str) -> str:
        reason = reason.strip()
        if len(reason) < MIN_REASON:
            raise ActionError("Give a reason")
        return reason[:500]

    def action(self, person: Person, action_id: str, step: str, reason: str) -> dict[str, Any] | None:
        if not Path(self.data.actions_db).exists():
            return None
        tracker = ActionTracker(sqlite_path=self.data.actions_db)
        current = tracker.get_action(action_id)
        if not current:
            return None
        if current["status"] == "closed":
            raise ActionError("This action is already closed")
        name = person[0]
        changes: dict[str, Any]
        if step == "start":
            reason = reason.strip() or "started"
            changes = {"status": "in_progress", "assigned_to": name, "acknowledged_at": _now()}
        else:
            reason = self._reason(reason)
            if step == "approve":
                if current.get("assigned_to") == name:
                    raise ActionError("Someone else must approve what you are doing yourself")
                changes = {"approved_by": name}
            elif step == "reject":
                changes = {"status": "closed", "acknowledged_at": current.get("acknowledged_at") or _now(),
                           "approval_required": False, "resolved_by": name, "closure_notes": f"Rejected: {reason}",
                           "verification_status": "WITHDRAWN"}
            else:
                if current.get("approval_required") and not current.get("approved_by"):
                    raise ActionError("This action needs approval before it can be closed")
                if current.get("approved_by") == name:
                    raise ActionError("The person who approved an action cannot also close it")
                changes = {"status": "closed", "acknowledged_at": current.get("acknowledged_at") or _now(),
                           "resolved_by": name, "closure_notes": reason}
        self._record(person, f"action_{step}", action_id, reason, title=(current.get("action_payload") or {}).get("title"))
        try:
            tracker.update_action(action_id, changed_by=name, change_reason=f"{step}: {reason}", **changes)
        except ValueError as e:
            raise ActionError(str(e)) from e
        return {"ok": True}

    def _registry(self) -> VisionModelRegistry:
        if not self.registry_dir:
            raise ActionError("This station has no model registry")
        return VisionModelRegistry(self.registry_dir)

    def retraining(self, person: Person, request_id: str, approve: bool, reason: str) -> dict[str, Any] | None:
        registry, reason = self._registry(), self._reason(reason)
        request = next((r for r in registry.requests() if r["id"] == request_id), None)
        if request is None:
            return None
        if request["status"] != "pending":
            raise ActionError(f"This request is already {request['status']}")
        self._record(person, "model_approve" if approve else "model_reject", request_id, reason)
        registry.decide(request_id, approve=approve, by=person[0], note=reason)
        return {"ok": True}

    def model(self, person: Person, recipe: str, step: str, reason: str) -> dict[str, Any] | None:
        registry, reason = self._registry(), self._reason(reason)
        if recipe not in registry.recipes():
            return None
        before = registry.manifest(recipe, "anomaly")
        if step == "promote" and not before.get("shadow"):
            raise ActionError("There is no shadow model to promote")
        if step == "rollback" and registry.earlier_version(recipe, "anomaly") is None:
            raise ActionError("There is no earlier model to go back to")
        self._record(person, f"model_{step}", f"{recipe}/anomaly", reason, version=before.get("shadow"),
                     replaces=before.get("active"))
        version = (registry.promote if step == "promote" else registry.rollback)(
            recipe, "anomaly", by=person[0], reason=reason)
        return {"ok": True, "version": version}

    def review(self, person: Person, camera: str, item: str, verdict: str, defect_type: str | None) -> dict[str, Any] | None:
        path = self._review_path(camera, item)
        if path is None:
            return None
        if verdict not in ("pass", "fail"):
            raise ActionError("Say whether the part was good or bad")
        meta = json.loads(path.read_text(encoding="utf-8"))
        if (meta.get("review") or {}).get("verdict"):
            raise ActionError("This part was already reviewed")
        self._record(person, "part_review", item, verdict, cycle_id=meta.get("cycle_id"), image_sha256=meta.get("image_sha"),
                     model_verdict=meta.get("model_verdict"))
        meta["review"] = {"verdict": verdict, "defect_type": defect_type if verdict == "fail" else None,
                          "reviewer": person[0]}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return {"ok": True}

    def baseline(self, person: Person, machine: str, process_change: bool, reason: str) -> dict[str, Any] | None:
        if self.data.store is None or machine not in self.data.store.machines():
            return None
        reason = self._reason(reason)
        self._record(person, "baseline_reset", machine, reason, process_change=process_change)
        self.data.store.reset_baseline(machine, kind=PROCESS_CHANGE if process_change else MAINTENANCE, reason=reason,
                                       by=person[0])
        return {"ok": True}


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
