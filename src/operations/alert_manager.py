from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


@dataclass
class AlertRecord:
    alert_code: str
    alert_title: str
    alert_level: str
    escalation_stage: str
    response_deadline: str
    source_window: str
    evidence_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_code": self.alert_code,
            "alert_title": self.alert_title,
            "alert_level": self.alert_level,
            "escalation_stage": self.escalation_stage,
            "response_deadline": self.response_deadline,
            "source_window": self.source_window,
            "evidence_summary": self.evidence_summary,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_windows(rca_payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "trends" in rca_payload and "anomalies" in rca_payload:
        return [("current", rca_payload)]
    ordered = ["last_hour", "last_shift", "last_24h", "all_time"]
    return [(k, rca_payload[k]) for k in ordered if isinstance(rca_payload.get(k), dict)]


def _deadline_in(minutes: int) -> str:
    return (_now() + timedelta(minutes=int(minutes))).isoformat()


def _trend_ratio(window_payload: dict[str, Any]) -> float:
    trends = window_payload.get("trends", {}) or {}
    per_hour = trends.get("defects_per_hour", []) or []
    rolling = trends.get("rolling_mean", []) or []
    if not per_hour or not rolling:
        return 0.0
    latest_count = float(per_hour[-1].get("defect_count", 0.0) or 0.0)
    latest_roll = float(rolling[-1].get("rolling_mean", 0.0) or 0.0)
    if latest_roll <= 1e-9:
        return 0.0
    return (latest_count - latest_roll) / latest_roll


def _critical_anomaly_alerts(rca_payload: dict[str, Any]) -> list[AlertRecord]:
    """Rule 1: Critical anomalies -> immediate alert."""
    alerts: list[AlertRecord] = []
    for window, payload in _extract_windows(rca_payload):
        anomalies = payload.get("anomalies", {}) or {}
        spikes = anomalies.get("spikes", []) or []
        threshold = float(anomalies.get("zscore_threshold", 2.0) or 2.0)
        if not spikes:
            continue

        max_z = max(float(s.get("zscore", 0.0) or 0.0) for s in spikes)
        # Critical if clearly above configured threshold margin.
        if max_z < (threshold * 1.5):
            continue

        alerts.append(
            AlertRecord(
                alert_code="ANOMALY_CRITICAL",
                alert_title="Critical anomaly pattern detected",
                alert_level="critical",
                escalation_stage="immediate_response",
                response_deadline=_deadline_in(30),
                source_window=window,
                evidence_summary={
                    "spike_count": int(len(spikes)),
                    "max_zscore": round(max_z, 4),
                    "threshold": threshold,
                    "rule": "max_zscore >= 1.5 * threshold",
                },
            )
        )
    return alerts


def _persistent_trend_warnings(rca_payload: dict[str, Any]) -> list[AlertRecord]:
    """Rule 2: Persistent trend increase -> warning."""
    windows = {k: v for k, v in _extract_windows(rca_payload)}

    ratio_hour = _trend_ratio(windows.get("last_hour", {}))
    ratio_24h = _trend_ratio(windows.get("last_24h", {}))

    # Persistent increase if both windows trend upward above 10%.
    if ratio_hour >= 0.10 and ratio_24h >= 0.10:
        return [
            AlertRecord(
                alert_code="TREND_PERSISTENT_WARNING",
                alert_title="Persistent defect trend increase",
                alert_level="warning",
                escalation_stage="stage_1_monitoring",
                response_deadline=_deadline_in(240),
                source_window="last_hour+last_24h",
                evidence_summary={
                    "last_hour_delta_ratio": round(ratio_hour, 4),
                    "last_24h_delta_ratio": round(ratio_24h, 4),
                    "rule": "last_hour>=10% and last_24h>=10%",
                },
            )
        ]
    return []


def load_unresolved_actions(sqlite_path: str = "data/events/actions.db") -> list[dict[str, Any]]:
    """Load unresolved actions from tracker DB for escalation rule."""
    if not Path(sqlite_path).exists():
        return []
    with sqlite3.connect(sqlite_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT action_id, timestamp, rca_reference, assigned_role, status, version, updated_at,
                   verification_status, target_value, verified_at, action_payload
            FROM tracked_actions
            WHERE status IN ('open', 'in_progress')
            ORDER BY datetime(timestamp) DESC
            """
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "action_id": r["action_id"],
                "timestamp": r["timestamp"],
                "rca_reference": r["rca_reference"],
                "assigned_role": r["assigned_role"],
                "status": r["status"],
                "version": int(r["version"]),
                "updated_at": r["updated_at"],
                "verification_status": r["verification_status"],
                "target_value": r["target_value"],
                "verified_at": r["verified_at"],
                "action_payload": json.loads(r["action_payload"]),
            }
        )
    return out


def _repeated_unresolved_escalations(unresolved_actions: list[dict[str, Any]]) -> list[AlertRecord]:
    """Rule 3: Repeated unresolved actions -> escalation."""
    if not unresolved_actions:
        return []

    # Group unresolved actions by RCA reference (industrial audit-friendly key).
    grouped: dict[str, list[dict[str, Any]]] = {}
    for a in unresolved_actions:
        key = str(a.get("rca_reference", "unknown"))
        grouped.setdefault(key, []).append(a)

    alerts: list[AlertRecord] = []
    for rca_ref, items in grouped.items():
        count = len(items)
        if count < 3:
            continue

        oldest_ts = min((str(i.get("timestamp", "")) for i in items), default="")
        alerts.append(
            AlertRecord(
                alert_code="UNRESOLVED_ACTION_ESCALATION",
                alert_title="Repeated unresolved actions require escalation",
                alert_level="high",
                escalation_stage="stage_2_management_escalation",
                response_deadline=_deadline_in(120),
                source_window="action_tracker",
                evidence_summary={
                    "rca_reference": rca_ref,
                    "unresolved_action_count": int(count),
                    "oldest_open_timestamp": oldest_ts,
                    "rule": "unresolved actions grouped by RCA reference >= 3",
                },
            )
        )
    return alerts


def _actuator_failures(unresolved_actions: list[dict[str, Any]]) -> list[AlertRecord]:
    alerts: list[AlertRecord] = []
    for action in unresolved_actions:
        verification_status = str(action.get("verification_status", "") or "").upper()
        if verification_status not in {"FAILED", "TIMEOUT"}:
            continue
        alerts.append(
            AlertRecord(
                alert_code="ACTUATOR_FAILURE",
                alert_title="Actuator command failed verification",
                alert_level="critical",
                escalation_stage="immediate_response",
                response_deadline=_deadline_in(15),
                source_window="action_verifier",
                evidence_summary={
                    "action_id": action.get("action_id"),
                    "verification_status": verification_status,
                    "rca_reference": action.get("rca_reference"),
                },
            )
        )
    return alerts


def build_alerts(
    rca_payload: dict[str, Any],
    unresolved_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Production-ready structured industrial alert generation."""
    unresolved_actions = unresolved_actions or []

    alerts = []
    alerts.extend(_critical_anomaly_alerts(rca_payload))
    alerts.extend(_persistent_trend_warnings(rca_payload))
    alerts.extend(_repeated_unresolved_escalations(unresolved_actions))
    alerts.extend(_actuator_failures(unresolved_actions))

    records = [a.to_dict() for a in alerts]
    level_rank = {"critical": 3, "high": 2, "warning": 1, "low": 0}
    records.sort(key=lambda x: level_rank.get(x.get("alert_level", "low"), 0), reverse=True)

    return {
        "alerts": records,
        "summary": {
            "total_alerts": len(records),
            "by_level": {
                "critical": sum(1 for r in records if r.get("alert_level") == "critical"),
                "high": sum(1 for r in records if r.get("alert_level") == "high"),
                "warning": sum(1 for r in records if r.get("alert_level") == "warning"),
                "low": sum(1 for r in records if r.get("alert_level") == "low"),
            },
        },
        "audit": {
            "generated_at_utc": _now().isoformat(),
            "rules_applied": [
                "critical_anomaly_immediate_alert",
                "persistent_trend_warning",
                "repeated_unresolved_actions_escalation",
                "actuator_failure_critical_alert",
            ],
            "schema_version": "1.0.0",
        },
    }


if __name__ == "__main__":
    sample_rca = {
        "last_hour": {
            "window_start": "2026-03-07T00:00:00+00:00",
            "window_end": "2026-03-07T01:00:00+00:00",
            "trends": {
                "defects_per_hour": [
                    {"hour": "2026-03-07 00:00:00", "defect_count": 40},
                    {"hour": "2026-03-07 01:00:00", "defect_count": 80},
                ],
                "rolling_mean": [
                    {"hour": "2026-03-07 00:00:00", "rolling_mean": 35.0},
                    {"hour": "2026-03-07 01:00:00", "rolling_mean": 50.0},
                ],
            },
            "anomalies": {
                "zscore_threshold": 2.0,
                "spikes": [
                    {"hour": "2026-03-07 01:00:00", "defect_count": 80, "zscore": 3.4},
                    {"hour": "2026-03-07 00:00:00", "defect_count": 70, "zscore": 3.0},
                ],
            },
        },
        "last_24h": {
            "trends": {
                "defects_per_hour": [
                    {"hour": "2026-03-06 23:00:00", "defect_count": 62},
                    {"hour": "2026-03-07 00:00:00", "defect_count": 79},
                ],
                "rolling_mean": [
                    {"hour": "2026-03-06 23:00:00", "rolling_mean": 54.0},
                    {"hour": "2026-03-07 00:00:00", "rolling_mean": 60.0},
                ],
            }
        },
    }

    sample_unresolved = [
        {"rca_reference": "rca:last_hour:2026-03-07T01:00:00Z", "status": "open", "timestamp": "2026-03-07T00:10:00+00:00"},
        {"rca_reference": "rca:last_hour:2026-03-07T01:00:00Z", "status": "in_progress", "timestamp": "2026-03-07T00:20:00+00:00"},
        {"rca_reference": "rca:last_hour:2026-03-07T01:00:00Z", "status": "open", "timestamp": "2026-03-07T00:30:00+00:00"},
    ]

    print(json.dumps(build_alerts(sample_rca, sample_unresolved), indent=2))