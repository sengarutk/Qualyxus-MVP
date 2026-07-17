from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_STATUSES = {"open", "in_progress", "closed"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TrackedAction:
    action_id: str
    timestamp: str
    rca_reference: str
    assigned_role: str
    assigned_to: str | None
    status: str
    acknowledged_at: str | None
    resolved_by: str | None
    approval_required: bool
    approved_by: str | None
    closure_notes: str | None
    verification_status: str | None
    target_value: float | None
    verified_at: str | None
    version: int
    action_payload: dict[str, Any]


class ActionTracker:
    """Industrial action tracking with SQLite storage and versioned history."""

    def __init__(self, sqlite_path: str = "data/events/actions.db"):
        self.sqlite_path = sqlite_path
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracked_actions (
                    action_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    rca_reference TEXT NOT NULL,
                    assigned_role TEXT NOT NULL,
                    assigned_to TEXT,
                    status TEXT NOT NULL,
                    acknowledged_at TEXT,
                    resolved_by TEXT,
                    approval_required INTEGER NOT NULL DEFAULT 0,
                    approved_by TEXT,
                    closure_notes TEXT,
                    verification_status TEXT,
                    target_value REAL,
                    verified_at TEXT,
                    version INTEGER NOT NULL,
                    action_payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tracked_action_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    changed_at TEXT NOT NULL,
                    changed_by TEXT,
                    status TEXT NOT NULL,
                    assigned_role TEXT NOT NULL,
                    assigned_to TEXT,
                    acknowledged_at TEXT,
                    resolved_by TEXT,
                    approval_required INTEGER NOT NULL DEFAULT 0,
                    approved_by TEXT,
                    closure_notes TEXT,
                    verification_status TEXT,
                    target_value REAL,
                    verified_at TEXT,
                    action_payload TEXT NOT NULL,
                    change_reason TEXT,
                    UNIQUE(action_id, version)
                )
                """
            )

            self._migrate_schema(conn)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_status ON tracked_actions(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_role ON tracked_actions(assigned_role)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON tracked_actions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_versions_action_id ON tracked_action_versions(action_id)")
            conn.commit()

    @staticmethod
    def _migrate_schema(conn: sqlite3.Connection) -> None:
        """Backward-compatible schema migration for existing SQLite files."""

        def add_missing_columns(table: str, required_cols: dict[str, str]) -> None:
            current_cols = {
                r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for col, col_type in required_cols.items():
                if col not in current_cols:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")

        add_missing_columns(
            "tracked_actions",
            {
                "assigned_to": "TEXT",
                "acknowledged_at": "TEXT",
                "resolved_by": "TEXT",
                "approval_required": "INTEGER NOT NULL DEFAULT 0",
                "approved_by": "TEXT",
                "verification_status": "TEXT",
                "target_value": "REAL",
                "verified_at": "TEXT",
            },
        )
        add_missing_columns(
            "tracked_action_versions",
            {
                "assigned_to": "TEXT",
                "acknowledged_at": "TEXT",
                "resolved_by": "TEXT",
                "approval_required": "INTEGER NOT NULL DEFAULT 0",
                "approved_by": "TEXT",
                "verification_status": "TEXT",
                "target_value": "REAL",
                "verified_at": "TEXT",
            },
        )

    def _insert_version(
        self,
        conn: sqlite3.Connection,
        *,
        action_id: str,
        version: int,
        status: str,
        assigned_role: str,
        assigned_to: str | None,
        acknowledged_at: str | None,
        resolved_by: str | None,
        approval_required: bool,
        approved_by: str | None,
        closure_notes: str | None,
        verification_status: str | None,
        target_value: float | None,
        verified_at: str | None,
        action_payload: dict[str, Any],
        changed_by: str | None,
        change_reason: str | None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO tracked_action_versions (
                action_id, version, changed_at, changed_by,
                status, assigned_role, assigned_to, acknowledged_at,
                resolved_by, approval_required, approved_by,
                closure_notes, verification_status, target_value, verified_at,
                action_payload, change_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                action_id,
                version,
                _utc_now(),
                changed_by,
                status,
                assigned_role,
                assigned_to,
                acknowledged_at,
                resolved_by,
                1 if approval_required else 0,
                approved_by,
                closure_notes,
                verification_status,
                target_value,
                verified_at,
                json.dumps(action_payload, ensure_ascii=False),
                change_reason,
            ),
        )

    def issue_action(
        self,
        *,
        rca_reference: str,
        assigned_role: str,
        assigned_to: str | None = None,
        action_payload: dict[str, Any],
        status: str = "open",
        approval_required: bool = False,
        closure_notes: str | None = None,
        verification_status: str | None = "PENDING",
        target_value: float | None = None,
        verified_at: str | None = None,
        action_id: str | None = None,
        changed_by: str | None = None,
    ) -> dict[str, Any]:
        """Create a new tracked action and version 1 history entry."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Must be one of {sorted(VALID_STATUSES)}")

        action_id = action_id or uuid.uuid4().hex
        timestamp = _utc_now()
        version = 1

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tracked_actions (
                    action_id, timestamp, rca_reference, assigned_role, assigned_to,
                    status, acknowledged_at, resolved_by, approval_required, approved_by,
                    closure_notes, verification_status, target_value, verified_at,
                    version, action_payload, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    timestamp,
                    rca_reference,
                    assigned_role,
                    assigned_to,
                    status,
                    None,
                    None,
                    1 if approval_required else 0,
                    None,
                    closure_notes,
                    verification_status,
                    target_value,
                    verified_at,
                    version,
                    json.dumps(action_payload, ensure_ascii=False),
                    timestamp,
                ),
            )

            self._insert_version(
                conn,
                action_id=action_id,
                version=version,
                status=status,
                assigned_role=assigned_role,
                assigned_to=assigned_to,
                acknowledged_at=None,
                resolved_by=None,
                approval_required=approval_required,
                approved_by=None,
                closure_notes=closure_notes,
                verification_status=verification_status,
                target_value=target_value,
                verified_at=verified_at,
                action_payload=action_payload,
                changed_by=changed_by,
                change_reason="initial_issue",
            )
            conn.commit()

        return self.get_action(action_id)

    def update_action(
        self,
        action_id: str,
        *,
        status: str | None = None,
        assigned_role: str | None = None,
        assigned_to: str | None = None,
        acknowledged_at: str | None = None,
        resolved_by: str | None = None,
        approval_required: bool | None = None,
        approved_by: str | None = None,
        closure_notes: str | None = None,
        verification_status: str | None = None,
        target_value: float | None = None,
        verified_at: str | None = None,
        action_payload: dict[str, Any] | None = None,
        changed_by: str | None = None,
        change_reason: str | None = None,
    ) -> dict[str, Any]:
        """Update current action row and append a new version history record."""
        current = self.get_action(action_id)
        if not current:
            raise KeyError(f"Action '{action_id}' not found")

        new_status = status or current["status"]
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Must be one of {sorted(VALID_STATUSES)}")

        new_assigned = assigned_role or current["assigned_role"]
        new_assigned_to = assigned_to if assigned_to is not None else current.get("assigned_to")
        new_acknowledged = acknowledged_at if acknowledged_at is not None else current.get("acknowledged_at")
        new_approval_required = (
            bool(approval_required)
            if approval_required is not None
            else bool(current.get("approval_required", False))
        )
        new_approved_by = approved_by if approved_by is not None else current.get("approved_by")

        if new_status == "closed" and not new_acknowledged:
            raise ValueError("Action cannot be closed until it has been acknowledged.")
        if new_status == "closed" and new_approval_required and not new_approved_by:
            raise ValueError("Action requires approval before closing.")

        new_resolved_by = resolved_by if resolved_by is not None else current.get("resolved_by")
        if new_status == "closed" and not new_resolved_by:
            new_resolved_by = changed_by

        new_notes = closure_notes if closure_notes is not None else current.get("closure_notes")
        new_verification_status = verification_status if verification_status is not None else current.get("verification_status")
        new_target_value = target_value if target_value is not None else current.get("target_value")
        new_verified_at = verified_at if verified_at is not None else current.get("verified_at")
        new_payload = action_payload if action_payload is not None else (current.get("action_payload") or {})
        new_version = int(current["version"]) + 1
        updated_at = _utc_now()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE tracked_actions
                SET assigned_role = ?, assigned_to = ?, status = ?, acknowledged_at = ?,
                    resolved_by = ?, approval_required = ?, approved_by = ?, closure_notes = ?,
                    verification_status = ?, target_value = ?, verified_at = ?,
                    version = ?, action_payload = ?, updated_at = ?
                WHERE action_id = ?
                """,
                (
                    new_assigned,
                    new_assigned_to,
                    new_status,
                    new_acknowledged,
                    new_resolved_by,
                    1 if new_approval_required else 0,
                    new_approved_by,
                    new_notes,
                    new_verification_status,
                    new_target_value,
                    new_verified_at,
                    new_version,
                    json.dumps(new_payload, ensure_ascii=False),
                    updated_at,
                    action_id,
                ),
            )

            self._insert_version(
                conn,
                action_id=action_id,
                version=new_version,
                status=new_status,
                assigned_role=new_assigned,
                assigned_to=new_assigned_to,
                acknowledged_at=new_acknowledged,
                resolved_by=new_resolved_by,
                approval_required=new_approval_required,
                approved_by=new_approved_by,
                closure_notes=new_notes,
                verification_status=new_verification_status,
                target_value=new_target_value,
                verified_at=new_verified_at,
                action_payload=new_payload,
                changed_by=changed_by,
                change_reason=change_reason,
            )
            conn.commit()

        return self.get_action(action_id)

    def acknowledge_action(self, action_id: str, user: str) -> dict[str, Any]:
        """Acknowledge assignment receipt by a user."""
        current = self.get_action(action_id)
        if not current:
            raise KeyError(f"Action '{action_id}' not found")

        if current.get("acknowledged_at"):
            # Idempotent behavior; keep existing ack timestamp.
            return current

        return self.update_action(
            action_id,
            assigned_to=user,
            acknowledged_at=_utc_now(),
            changed_by=user,
            change_reason="acknowledged",
        )

    def approve_action(self, action_id: str, user: str) -> dict[str, Any]:
        """Approve action when approval gate is required."""
        current = self.get_action(action_id)
        if not current:
            raise KeyError(f"Action '{action_id}' not found")

        if not bool(current.get("approval_required", False)):
            raise ValueError("Action does not require approval.")

        return self.update_action(
            action_id,
            approved_by=user,
            changed_by=user,
            change_reason="approved",
        )

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tracked_actions WHERE action_id = ?", (action_id,)).fetchone()
        if not row:
            return None
        return self._row_to_action_dict(row)

    def list_actions(
        self,
        *,
        status: str | None = None,
        assigned_role: str | None = None,
        rca_reference: str | None = None,
        from_ts: str | None = None,
        to_ts: str | None = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Audit query across current action state with filters."""
        clauses = ["1=1"]
        params: list[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status)
        if assigned_role:
            clauses.append("assigned_role = ?")
            params.append(assigned_role)
        if rca_reference:
            clauses.append("rca_reference = ?")
            params.append(rca_reference)
        if from_ts:
            clauses.append("timestamp >= ?")
            params.append(from_ts)
        if to_ts:
            clauses.append("timestamp <= ?")
            params.append(to_ts)

        params.append(int(limit))
        sql = f"""
            SELECT *
            FROM tracked_actions
            WHERE {' AND '.join(clauses)}
            ORDER BY datetime(timestamp) DESC
            LIMIT ?
        """

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._row_to_action_dict(r) for r in rows]

    def get_action_history(self, action_id: str) -> list[dict[str, Any]]:
        """Version history for a specific action (audit trail)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM tracked_action_versions
                WHERE action_id = ?
                ORDER BY version ASC
                """,
                (action_id,),
            ).fetchall()

        out = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "action_id": r["action_id"],
                    "version": r["version"],
                    "changed_at": r["changed_at"],
                    "changed_by": r["changed_by"],
                    "status": r["status"],
                    "assigned_role": r["assigned_role"],
                    "assigned_to": r["assigned_to"],
                    "acknowledged_at": r["acknowledged_at"],
                    "resolved_by": r["resolved_by"],
                    "approval_required": bool(r["approval_required"]),
                    "approved_by": r["approved_by"],
                    "closure_notes": r["closure_notes"],
                    "verification_status": r["verification_status"],
                    "target_value": r["target_value"],
                    "verified_at": r["verified_at"],
                    "action_payload": json.loads(r["action_payload"]),
                    "change_reason": r["change_reason"],
                }
            )
        return out

    def audit_summary(self) -> dict[str, Any]:
        """Quick high-level audit counters."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) AS c FROM tracked_actions").fetchone()["c"]
            status_rows = conn.execute(
                "SELECT status, COUNT(*) AS c FROM tracked_actions GROUP BY status"
            ).fetchall()
            role_rows = conn.execute(
                "SELECT assigned_role, COUNT(*) AS c FROM tracked_actions GROUP BY assigned_role"
            ).fetchall()

        return {
            "total_actions": int(total),
            "by_status": {r["status"]: int(r["c"]) for r in status_rows},
            "by_assigned_role": {r["assigned_role"]: int(r["c"]) for r in role_rows},
        }

    @staticmethod
    def _row_to_action_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "action_id": row["action_id"],
            "timestamp": row["timestamp"],
            "rca_reference": row["rca_reference"],
            "assigned_role": row["assigned_role"],
            "assigned_to": row["assigned_to"],
            "status": row["status"],
            "acknowledged_at": row["acknowledged_at"],
            "resolved_by": row["resolved_by"],
            "approval_required": bool(row["approval_required"]),
            "approved_by": row["approved_by"],
            "closure_notes": row["closure_notes"],
            "verification_status": row["verification_status"],
            "target_value": row["target_value"],
            "verified_at": row["verified_at"],
            "version": int(row["version"]),
            "action_payload": json.loads(row["action_payload"]),
            "updated_at": row["updated_at"],
        }


if __name__ == "__main__":
    tracker = ActionTracker(sqlite_path="data/events/actions.db")

    issued = tracker.issue_action(
        rca_reference="rca:last_hour:2026-03-07T01:30:00Z",
        assigned_role="maintenance",
        assigned_to="maint:lead-01",
        action_payload={
            "title": "Inspect Machine A optical alignment",
            "action_type": "corrective",
            "severity_score": 72.5,
        },
        approval_required=True,
        changed_by="system:auto",
    )

    acknowledged = tracker.acknowledge_action(issued["action_id"], user="maint:lead-01")

    approved = tracker.approve_action(issued["action_id"], user="qa:reviewer-02")

    updated = tracker.update_action(issued["action_id"], status="in_progress", changed_by="maint:lead-01", change_reason="work_started")

    closed = tracker.update_action(
        issued["action_id"],
        status="closed",
        resolved_by="maint:lead-01",
        closure_notes="Camera mount recalibrated; no defects in next 2h sample.",
        changed_by="qa:reviewer-02",
        change_reason="verified_closed",
    )

    print(
        json.dumps(
            {
                "issued": issued,
                "acknowledged": acknowledged,
                "approved": approved,
                "updated": updated,
                "closed": closed,
                "history": tracker.get_action_history(issued["action_id"]),
                "open_actions": tracker.list_actions(status="open", limit=20),
                "audit_summary": tracker.audit_summary(),
            },
            indent=2,
        )
    )