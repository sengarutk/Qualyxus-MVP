from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.root_cause import run_root_cause


DEFAULT_EVENTS_DB = "data/events/events.db"
DEFAULT_ACTIONS_DB = "data/events/actions.db"


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@dataclass
class TimelineWindowResult:
    window: str
    event_count: int
    event_sequence: list[dict[str, Any]]
    time_gaps: list[dict[str, Any]]
    transitions: dict[str, Any]
    linked_anomalies: dict[str, Any]
    linked_correlations: dict[str, Any]
    linked_actions: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "event_count": self.event_count,
            "event_sequence": self.event_sequence,
            "time_gaps": self.time_gaps,
            "transitions": self.transitions,
            "linked_anomalies": self.linked_anomalies,
            "linked_correlations": self.linked_correlations,
            "linked_actions": self.linked_actions,
        }


class ProcessTimelineReconstructor:
    """Reconstruct ordered process timelines for RCA windows and link operational context."""

    def __init__(
        self,
        events_db_path: str = DEFAULT_EVENTS_DB,
        actions_db_path: str = DEFAULT_ACTIONS_DB,
    ):
        self.events_db_path = events_db_path
        self.actions_db_path = actions_db_path

    def _load_events(self) -> pd.DataFrame:
        db_path = Path(self.events_db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Events DB not found: {db_path}")

        query = """
            SELECT event_id, timestamp, source, defect_type, confidence, severity,
                   operator_id, machine_id, shift_id
            FROM defect_events
        """
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(query, conn)

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.dropna(subset=["timestamp"]).copy()
        for col in ["source", "defect_type", "operator_id", "machine_id", "shift_id"]:
            df[col] = df[col].fillna("unknown").astype(str)
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)
        df["severity"] = pd.to_numeric(df["severity"], errors="coerce").fillna(0)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _load_actions(self) -> list[dict[str, Any]]:
        db_path = Path(self.actions_db_path)
        if not db_path.exists():
            return []

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT action_id, timestamp, rca_reference, assigned_role, assigned_to,
                       status, acknowledged_at, resolved_by, approval_required, approved_by,
                       closure_notes, action_payload, updated_at
                FROM tracked_actions
                ORDER BY datetime(timestamp) DESC
                """
            ).fetchall()

        out = []
        for r in rows:
            out.append(
                {
                    "action_id": r["action_id"],
                    "timestamp": r["timestamp"],
                    "rca_reference": r["rca_reference"],
                    "assigned_role": r["assigned_role"],
                    "assigned_to": r["assigned_to"],
                    "status": r["status"],
                    "acknowledged_at": r["acknowledged_at"],
                    "resolved_by": r["resolved_by"],
                    "approval_required": bool(r["approval_required"]),
                    "approved_by": r["approved_by"],
                    "closure_notes": r["closure_notes"],
                    "action_payload": json.loads(r["action_payload"]),
                    "updated_at": r["updated_at"],
                }
            )
        return out

    @staticmethod
    def _window_keys(rca_payload: dict[str, Any]) -> list[str]:
        ordered = ["last_hour", "last_shift", "last_24h", "all_time"]
        return [k for k in ordered if isinstance(rca_payload.get(k), dict)]

    @staticmethod
    def _build_event_sequence(window_events: pd.DataFrame) -> list[dict[str, Any]]:
        if window_events.empty:
            return []
        seq = []
        for idx, r in enumerate(window_events.itertuples(index=False), start=1):
            seq.append(
                {
                    "sequence_index": idx,
                    "event_id": str(r.event_id),
                    "timestamp": _iso(r.timestamp.to_pydatetime()),
                    "defect_type": str(r.defect_type),
                    "machine_id": str(r.machine_id),
                    "operator_id": str(r.operator_id),
                    "shift_id": str(r.shift_id),
                    "source": str(r.source),
                    "confidence": round(float(r.confidence), 6),
                    "severity": int(r.severity),
                }
            )
        return seq

    @staticmethod
    def _build_time_gaps(window_events: pd.DataFrame) -> list[dict[str, Any]]:
        if len(window_events) < 2:
            return []

        w = window_events.sort_values("timestamp").copy()
        w["next_timestamp"] = w["timestamp"].shift(-1)
        w = w.iloc[:-1].copy()
        w["gap_seconds"] = (w["next_timestamp"] - w["timestamp"]).dt.total_seconds()

        # Deterministic long-gap threshold (5 minutes).
        threshold = 300.0
        gaps = []
        for r in w.itertuples(index=False):
            gaps.append(
                {
                    "from_event_id": str(r.event_id),
                    "from_timestamp": _iso(r.timestamp.to_pydatetime()),
                    "to_timestamp": _iso(r.next_timestamp.to_pydatetime()),
                    "gap_seconds": round(float(r.gap_seconds), 3),
                    "is_long_gap": bool(float(r.gap_seconds) > threshold),
                }
            )
        return gaps

    @staticmethod
    def _build_transitions(window_events: pd.DataFrame) -> dict[str, Any]:
        if len(window_events) < 2:
            return {"defect_type_transitions": [], "machine_transitions": []}

        w = window_events.sort_values("timestamp").copy()

        def transition_counts(col: str) -> list[dict[str, Any]]:
            curr = w[col].astype(str)
            nxt = curr.shift(-1)
            trans = pd.DataFrame({"from": curr.iloc[:-1], "to": nxt.iloc[:-1]})
            grouped = trans.groupby(["from", "to"]).size().reset_index(name="count")
            grouped = grouped.sort_values("count", ascending=False)
            return grouped.to_dict(orient="records")

        return {
            "defect_type_transitions": transition_counts("defect_type"),
            "machine_transitions": transition_counts("machine_id"),
        }

    @staticmethod
    def _link_anomalies(window_payload: dict[str, Any], window_events: pd.DataFrame) -> dict[str, Any]:
        anomalies = (window_payload.get("anomalies", {}) or {}).get("spikes", []) or []
        if not anomalies or window_events.empty:
            return {"spike_hours": [], "linked_event_ids": []}

        spike_hours = {str(a.get("hour", "")) for a in anomalies}
        # Event hour format aligns with root_cause format.
        hours = window_events["timestamp"].dt.strftime("%Y-%m-%d %H:00:00")
        linked_ids = window_events.loc[hours.isin(spike_hours), "event_id"].astype(str).tolist()

        return {
            "spike_hours": sorted([h for h in spike_hours if h]),
            "linked_event_ids": linked_ids,
            "spike_count": len(anomalies),
        }

    @staticmethod
    def _link_correlations(window_payload: dict[str, Any]) -> dict[str, Any]:
        corr = window_payload.get("correlations", {}) or {}

        def top(dim: str) -> dict[str, Any] | None:
            rows = corr.get(dim, []) or []
            if not rows:
                return None
            r = rows[0]
            return {
                "dimension": dim,
                "value": str(r.get(dim, "unknown")),
                "lift_ratio": round(float(r.get("lift_ratio", 1.0) or 1.0), 4),
                "defect_rate": round(float(r.get("defect_rate", 0.0) or 0.0), 6),
            }

        return {
            "top_machine": top("machine_id"),
            "top_operator": top("operator_id"),
            "top_shift": top("shift_id"),
            "top_defect_type": top("defect_type"),
        }

    @staticmethod
    def _link_actions(
        actions: list[dict[str, Any]],
        *,
        window_name: str,
        window_start: datetime | None,
        window_end: datetime | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for a in actions:
            rca_ref = str(a.get("rca_reference", "") or "")
            ts = _parse_ts(a.get("timestamp"))

            linked = False
            link_basis = []

            if window_name in rca_ref:
                linked = True
                link_basis.append("rca_reference_match")

            if ts and window_start and window_end and window_start <= ts < window_end:
                linked = True
                link_basis.append("timestamp_in_window")

            if not linked:
                continue

            out.append(
                {
                    "action_id": a.get("action_id"),
                    "status": a.get("status"),
                    "timestamp": a.get("timestamp"),
                    "assigned_role": a.get("assigned_role"),
                    "assigned_to": a.get("assigned_to"),
                    "rca_reference": rca_ref,
                    "link_basis": link_basis,
                }
            )

        return out

    def reconstruct(self, rca_payload: dict[str, Any]) -> dict[str, Any]:
        events = self._load_events()
        actions = self._load_actions()

        windows_out: dict[str, Any] = {}
        for window_name in self._window_keys(rca_payload):
            wp = rca_payload.get(window_name, {}) or {}
            ws = _parse_ts(wp.get("window_start"))
            we = _parse_ts(wp.get("window_end"))

            if ws and we and not events.empty:
                w_events = events[(events["timestamp"] >= ws) & (events["timestamp"] < we)].copy()
            else:
                # all_time or missing explicit boundaries
                w_events = events.copy()

            sequence = self._build_event_sequence(w_events)
            gaps = self._build_time_gaps(w_events)
            transitions = self._build_transitions(w_events)

            linked_anomalies = self._link_anomalies(wp, w_events)
            linked_correlations = self._link_correlations(wp)
            linked_actions = self._link_actions(actions, window_name=window_name, window_start=ws, window_end=we)

            windows_out[window_name] = TimelineWindowResult(
                window=window_name,
                event_count=int(len(w_events)),
                event_sequence=sequence,
                time_gaps=gaps,
                transitions=transitions,
                linked_anomalies=linked_anomalies,
                linked_correlations=linked_correlations,
                linked_actions=linked_actions,
            ).to_dict()

        return {
            "windows": windows_out,
            "audit": {
                "generated_at_utc": _iso(datetime.now(timezone.utc)),
                "events_db_path": self.events_db_path,
                "actions_db_path": self.actions_db_path,
                "schema_version": "1.0.0",
            },
        }


def build_process_timeline(
    rca_payload: dict[str, Any],
    *,
    events_db_path: str = DEFAULT_EVENTS_DB,
    actions_db_path: str = DEFAULT_ACTIONS_DB,
) -> dict[str, Any]:
    recon = ProcessTimelineReconstructor(events_db_path=events_db_path, actions_db_path=actions_db_path)
    return recon.reconstruct(rca_payload)


if __name__ == "__main__":
    rca = run_root_cause(sqlite_path=DEFAULT_EVENTS_DB)
    timeline = build_process_timeline(rca)
    print(json.dumps(timeline, indent=2))