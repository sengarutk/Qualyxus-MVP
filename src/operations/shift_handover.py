from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.root_cause import run_root_cause
from src.operations.alert_manager import build_alerts


DEFAULT_EVENTS_DB = "data/events/events.db"
DEFAULT_ACTIONS_DB = "data/events/actions.db"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _last_shift_window(reference_ts: datetime | None = None) -> tuple[datetime, datetime]:
    """
    Shift model (UTC):
    - A: 06:00-14:00
    - B: 14:00-22:00
    - C: 22:00-06:00
    Returns previous full shift window [start, end).
    """
    ts = (reference_ts or _utc_now()).astimezone(timezone.utc)
    h = ts.hour

    if 6 <= h < 14:
        current_start = ts.replace(hour=6, minute=0, second=0, microsecond=0)
    elif 14 <= h < 22:
        current_start = ts.replace(hour=14, minute=0, second=0, microsecond=0)
    elif h >= 22:
        current_start = ts.replace(hour=22, minute=0, second=0, microsecond=0)
    else:
        prev_day = ts - timedelta(days=1)
        current_start = prev_day.replace(hour=22, minute=0, second=0, microsecond=0)

    last_end = current_start
    last_start = last_end - timedelta(hours=8)
    return last_start, last_end


@dataclass
class ShiftHandoverInputs:
    events_db_path: str = DEFAULT_EVENTS_DB
    actions_db_path: str = DEFAULT_ACTIONS_DB


class ShiftHandoverGenerator:
    """Generate structured shift handover with event/action/alert status and narrative."""

    def __init__(self, config: ShiftHandoverInputs | None = None):
        self.config = config or ShiftHandoverInputs()

    def _load_last_shift_events(self, start: datetime, end: datetime) -> pd.DataFrame:
        db_path = Path(self.config.events_db_path)
        if not db_path.exists():
            return pd.DataFrame()

        query = """
            SELECT event_id, timestamp, source, defect_type, confidence, severity,
                   operator_id, machine_id, shift_id
            FROM defect_events
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY timestamp ASC
        """
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(query, conn, params=(_iso(start), _iso(end)))

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).copy()
        for col in ["source", "defect_type", "operator_id", "machine_id", "shift_id"]:
            df[col] = df[col].fillna("unknown").astype(str)
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0)
        return df

    def _summarize_events(self, events: pd.DataFrame) -> dict[str, Any]:
        if events.empty:
            return {
                "event_count": 0,
                "top_defect_types": [],
                "top_machines": [],
                "avg_confidence": 0.0,
                "avg_severity": 0.0,
                "recent_events": [],
            }

        top_defects = (
            events.groupby("defect_type").size().sort_values(ascending=False).head(5).reset_index(name="count")
        )
        top_machines = (
            events.groupby("machine_id").size().sort_values(ascending=False).head(5).reset_index(name="count")
        )

        recent = events.sort_values("timestamp", ascending=False).head(10)
        recent_events = [
            {
                "event_id": str(r.event_id),
                "timestamp": _iso(r.timestamp.to_pydatetime()),
                "defect_type": str(r.defect_type),
                "machine_id": str(r.machine_id),
                "severity": int(r.severity),
            }
            for r in recent.itertuples(index=False)
        ]

        return {
            "event_count": int(len(events)),
            "top_defect_types": top_defects.to_dict(orient="records"),
            "top_machines": top_machines.to_dict(orient="records"),
            "avg_confidence": round(float(events["confidence"].mean()), 4),
            "avg_severity": round(float(events["severity"].mean()), 4),
            "recent_events": recent_events,
        }

    def _load_open_actions(self) -> list[dict[str, Any]]:
        db_path = Path(self.config.actions_db_path)
        if not db_path.exists():
            return []

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT action_id, timestamp, rca_reference, assigned_role, assigned_to,
                       status, approval_required, approved_by, acknowledged_at
                FROM tracked_actions
                WHERE status IN ('open', 'in_progress')
                ORDER BY datetime(timestamp) DESC
                """
            ).fetchall()

        return [
            {
                "action_id": r["action_id"],
                "timestamp": r["timestamp"],
                "rca_reference": r["rca_reference"],
                "assigned_role": r["assigned_role"],
                "assigned_to": r["assigned_to"],
                "status": r["status"],
                "approval_required": bool(r["approval_required"]),
                "approved_by": r["approved_by"],
                "acknowledged_at": r["acknowledged_at"],
            }
            for r in rows
        ]

    @staticmethod
    def _critical_alerts(rca_output: dict[str, Any], open_actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        unresolved = [
            {
                "rca_reference": a.get("rca_reference"),
                "status": a.get("status"),
                "timestamp": a.get("timestamp"),
            }
            for a in open_actions
        ]
        alerts = build_alerts(rca_output, unresolved_actions=unresolved).get("alerts", [])
        return [a for a in alerts if str(a.get("alert_level", "")).lower() in {"critical", "high"}]

    @staticmethod
    def _narrative(
        *,
        shift_start: datetime,
        shift_end: datetime,
        event_summary: dict[str, Any],
        open_actions: list[dict[str, Any]],
        critical_alerts: list[dict[str, Any]],
    ) -> str:
        top_defect = (event_summary.get("top_defect_types") or [{}])[0]
        defect_name = top_defect.get("defect_type", "none")
        defect_count = top_defect.get("count", 0)

        return (
            f"Shift handover ({_iso(shift_start)} to {_iso(shift_end)}): "
            f"{event_summary.get('event_count', 0)} events recorded. "
            f"Top defect '{defect_name}' occurred {defect_count} times. "
            f"Open actions pending: {len(open_actions)}. "
            f"Critical/high alerts requiring attention: {len(critical_alerts)}. "
            f"Incoming shift should prioritize unresolved high-risk alerts and action acknowledgements."
        )

    def generate_handover(self) -> dict[str, Any]:
        shift_start, shift_end = _last_shift_window()
        events = self._load_last_shift_events(shift_start, shift_end)
        event_summary = self._summarize_events(events)

        open_actions = self._load_open_actions()

        # RCA context used for alert linkage (includes last_shift window).
        rca_output = run_root_cause(sqlite_path=self.config.events_db_path)
        critical_alerts = self._critical_alerts(rca_output, open_actions)

        return {
            "handover_window": {
                "shift_start": _iso(shift_start),
                "shift_end": _iso(shift_end),
                "duration_hours": 8,
            },
            "last_shift_events": event_summary,
            "open_actions": {
                "count": len(open_actions),
                "items": open_actions,
            },
            "critical_alerts": {
                "count": len(critical_alerts),
                "items": critical_alerts,
            },
            "narrative_summary": self._narrative(
                shift_start=shift_start,
                shift_end=shift_end,
                event_summary=event_summary,
                open_actions=open_actions,
                critical_alerts=critical_alerts,
            ),
            "audit": {
                "generated_at_utc": _iso(_utc_now()),
                "events_db_path": self.config.events_db_path,
                "actions_db_path": self.config.actions_db_path,
                "schema_version": "1.0.0",
            },
        }


if __name__ == "__main__":
    generator = ShiftHandoverGenerator()
    print(json.dumps(generator.generate_handover(), indent=2))