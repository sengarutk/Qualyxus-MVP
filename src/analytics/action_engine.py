from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any



@dataclass
class SignalSnapshot:
    window: str
    window_start: str
    window_end: str
    anomaly_strength: float
    anomaly_event_count: int
    max_zscore: float
    correlation_lift: float
    raw_lift_ratio: float
    trend_change_ratio: float
    trend_direction: str
    strongest_dimension: str
    strongest_factor: str
    target_system: str
    classification_6m: str


def _classification_from_dimension(dimension: str) -> str:
    mapping = {
        "operator_id": "Man",
        "machine_id": "Machine",
        "shift_id": "Environment",
        "defect_type": "Material",
    }
    return mapping.get(dimension, "Method")


def _extract_windows(rca_payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if "trends" in rca_payload and "correlations" in rca_payload:
        return [("current", rca_payload)]

    ordered = ["last_hour", "last_shift", "last_24h", "all_time"]
    windows: list[tuple[str, dict[str, Any]]] = []
    for key in ordered:
        val = rca_payload.get(key)
        if isinstance(val, dict):
            windows.append((key, val))
    return windows


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _anomaly_strength(window_payload: dict[str, Any]) -> float:
    anomalies = window_payload.get("anomalies", {}) or {}
    spikes = anomalies.get("spikes", []) or []
    threshold = float(anomalies.get("zscore_threshold", 2.0) or 2.0)
    if not spikes:
        return 0.0
    max_z = max(float(s.get("zscore", 0.0) or 0.0) for s in spikes)
    # Normalized around threshold -> 0.5, ~2x threshold -> 1.0
    return _clip01(max_z / max(threshold * 2.0, 1e-9))


def _trend_metrics(window_payload: dict[str, Any]) -> tuple[float, str]:
    trends = window_payload.get("trends", {}) or {}
    per_hour = trends.get("defects_per_hour", []) or []
    rolling = trends.get("rolling_mean", []) or []
    if not per_hour or not rolling:
        return 0.0, "flat"

    latest = per_hour[-1]
    latest_roll = rolling[-1]
    count = float(latest.get("defect_count", 0.0) or 0.0)
    base = float(latest_roll.get("rolling_mean", 0.0) or 0.0)
    if base <= 1e-9:
        return 0.0, "flat"

    ratio = (count - base) / base
    direction = "up" if ratio > 0.03 else "down" if ratio < -0.03 else "flat"
    # Magnitude normalized at 100% delta -> 1.0
    magnitude = _clip01(abs(ratio))
    return magnitude, direction


def _target_map(dimension: str) -> str:
    if dimension == "machine_id":
        return "machine"
    if dimension in {"shift_id", "operator_id"}:
        return "operator"
    return "process"


def _correlation_metrics(window_payload: dict[str, Any]) -> tuple[float, str, str, str]:
    corr = window_payload.get("correlations", {}) or {}
    best_dimension = "defect_type"
    best_factor = "unknown"
    best_lift = 1.0

    for dim in ["machine_id", "shift_id", "operator_id", "defect_type"]:
        rows = corr.get(dim, []) or []
        if not rows:
            continue
        top = rows[0]
        lift = float(top.get("lift_ratio", 1.0) or 1.0)
        factor = str(top.get(dim, "unknown"))
        if lift > best_lift:
            best_lift = lift
            best_dimension = dim
            best_factor = factor

    # Lift=1.0 means no elevation over baseline.
    lift_strength = _clip01((best_lift - 1.0) / 2.0)
    return lift_strength, best_dimension, best_factor, _target_map(best_dimension)


def _snapshot(window: str, payload: dict[str, Any]) -> SignalSnapshot:
    anomalies = payload.get("anomalies", {}) or {}
    spikes = anomalies.get("spikes", []) or []
    max_z = max([float(s.get("zscore", 0.0) or 0.0) for s in spikes], default=0.0)

    a = _anomaly_strength(payload)
    t_mag, t_dir = _trend_metrics(payload)
    c, dim, factor, target = _correlation_metrics(payload)

    # c is normalized lift strength in [0,1], map back to representative lift ratio.
    raw_lift = 1.0 + 2.0 * c

    window_start = str(payload.get("window_start", "n/a"))
    window_end = str(payload.get("window_end", "n/a"))

    return SignalSnapshot(
        window=window,
        window_start=window_start,
        window_end=window_end,
        anomaly_strength=a,
        anomaly_event_count=len(spikes),
        max_zscore=max_z,
        correlation_lift=c,
        raw_lift_ratio=raw_lift,
        trend_change_ratio=t_mag,
        trend_direction=t_dir,
        strongest_dimension=dim,
        strongest_factor=factor,
        target_system=target,
        classification_6m=_classification_from_dimension(dim),
    )


def _urgency_from_score(score: float) -> str:
    if score >= 85:
        return "critical"
    if score >= 70:
        return "high"
    if score >= 50:
        return "medium"
    return "low"


def _resolution_time(action_type: str, urgency: str) -> str:
    matrix = {
        "containment": {"critical": "< 30 min", "high": "< 2 h", "medium": "< 8 h", "low": "< 24 h"},
        "corrective": {"critical": "< 4 h", "high": "< 1 shift", "medium": "< 24 h", "low": "< 72 h"},
        "preventive": {"critical": "< 24 h", "high": "< 3 days", "medium": "< 7 days", "low": "< 14 days"},
    }
    return matrix[action_type][urgency]


def _responsible_role(action_type: str, target_system: str) -> str:
    if action_type == "containment":
        return "production"
    if action_type == "corrective":
        if target_system == "machine":
            return "maintenance"
        if target_system == "operator":
            return "production"
        return "QA"
    return "QA"


def _score_containment(s: SignalSnapshot) -> float:
    # Immediate response driven primarily by anomaly and fast trend rise.
    trend_up = s.trend_change_ratio if s.trend_direction == "up" else 0.0
    return min(100.0, 100.0 * (0.65 * s.anomaly_strength + 0.25 * trend_up + 0.10 * s.correlation_lift))


def _score_corrective(s: SignalSnapshot) -> float:
    # Short-term fix driven primarily by strongest correlation lift.
    return min(100.0, 100.0 * (0.20 * s.anomaly_strength + 0.55 * s.correlation_lift + 0.25 * s.trend_change_ratio))


def _score_preventive(s: SignalSnapshot) -> float:
    # Long-term fix driven by sustained frequency trend + recurring correlation.
    return min(100.0, 100.0 * (0.15 * s.anomaly_strength + 0.35 * s.correlation_lift + 0.50 * s.trend_change_ratio))


def _action_record(action_type: str, score: float, s: SignalSnapshot) -> dict[str, Any]:
    urgency = _urgency_from_score(score)
    contributing_signals = ["anomaly_strength", "correlation_lift", "frequency_trends"]

    if action_type == "containment":
        root_factor = f"anomaly_strength={s.anomaly_strength:.3f}, trend_direction={s.trend_direction}"
        title = "Immediate containment response"
        explanation = (
            f"Spike detected due to {s.anomaly_event_count} anomaly events "
            f"and {s.raw_lift_ratio:.2f}x lift on {s.strongest_factor}."
        )
    elif action_type == "corrective":
        root_factor = f"{s.strongest_dimension}:{s.strongest_factor}, correlation_lift={s.raw_lift_ratio:.3f}"
        title = "Corrective root-cause fix"
        explanation = (
            f"Corrective fix prioritized from {s.raw_lift_ratio:.2f}x lift on {s.strongest_dimension}="
            f"{s.strongest_factor} with trend direction {s.trend_direction}."
        )
    else:
        root_factor = (
            f"trend_change_ratio={s.trend_change_ratio:.3f}, trend_direction={s.trend_direction}, "
            f"correlation_lift={s.raw_lift_ratio:.3f}"
        )
        title = "Preventive control hardening"
        explanation = (
            f"Preventive action derived from frequency trend {s.trend_direction} "
            f"({s.trend_change_ratio:.3f}) and sustained lift {s.raw_lift_ratio:.2f}x."
        )

    return {
        "title": title,
        "explanation": explanation,
        "window": s.window,
        "classification_6m": s.classification_6m,
        "action_type": action_type,
        "target_system": s.target_system,
        "root_cause_factor": root_factor,
        "urgency_level": urgency,
        "expected_resolution_time": _resolution_time(action_type, urgency),
        "responsible_role": _responsible_role(action_type, s.target_system),
        "severity_score": round(float(score), 2),
        "contributing_signals": contributing_signals,
        "data_window_used": {"start": s.window_start, "end": s.window_end},
        "evidence_summary": {
            "anomaly_event_count": int(s.anomaly_event_count),
            "max_zscore": round(float(s.max_zscore), 4),
            "correlation_lift_ratio": round(float(s.raw_lift_ratio), 4),
            "trend_change_ratio": round(float(s.trend_change_ratio), 4),
            "trend_direction": s.trend_direction,
            "strongest_dimension": s.strongest_dimension,
            "strongest_factor": s.strongest_factor,
        },
        "signal_basis": {
            "anomaly_strength": round(s.anomaly_strength, 4),
            "correlation_lift_strength": round(s.correlation_lift, 4),
            "trend_change_strength": round(s.trend_change_ratio, 4),
            "trend_direction": s.trend_direction,
        },
    }


def _rank(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: float(r.get("severity_score", 0.0)), reverse=True)


def build_actions(
    rca_payload: dict[str, Any],
    narrative_insights: dict[str, list[dict[str, Any]]] | list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build audit-ready actions from RCA signals.

    Decision logic is based only on:
    - anomaly strength
    - correlation lift
    - frequency trends

    narrative_insights is accepted for interface compatibility but not used in decision scoring.
    """
    _ = narrative_insights  # Explicitly unused by standards-constrained logic.

    containment_actions: list[dict[str, Any]] = []
    corrective_actions: list[dict[str, Any]] = []
    preventive_actions: list[dict[str, Any]] = []

    windows = _extract_windows(rca_payload)
    for window_name, payload in windows:
        s = _snapshot(window_name, payload)

        containment_actions.append(_action_record("containment", _score_containment(s), s))
        corrective_actions.append(_action_record("corrective", _score_corrective(s), s))
        preventive_actions.append(_action_record("preventive", _score_preventive(s), s))

    containment_actions = _rank(containment_actions)
    corrective_actions = _rank(corrective_actions)
    preventive_actions = _rank(preventive_actions)

    priority_actions = _rank(containment_actions + corrective_actions + preventive_actions)[:8]

    return {
        "containment_actions": containment_actions,
        "corrective_actions": corrective_actions,
        "preventive_actions": preventive_actions,
        "priority_actions": priority_actions,
        "audit": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_windows": [w for w, _ in windows],
            "decision_basis": ["anomaly_strength", "correlation_lift", "frequency_trends"],
            "schema_version": "1.0.0",
            "compliance_note": "No non-signal heuristic used for decision scoring.",
        },
    }


if __name__ == "__main__":
    sample_rca = {
        "last_hour": {
            "trends": {
                "defects_per_hour": [
                    {"hour": "2026-03-07 21:00:00", "defect_count": 40},
                    {"hour": "2026-03-07 22:00:00", "defect_count": 78},
                ],
                "rolling_mean": [
                    {"hour": "2026-03-07 21:00:00", "rolling_mean": 38.0},
                    {"hour": "2026-03-07 22:00:00", "rolling_mean": 45.0},
                ],
            },
            "correlations": {
                "machine_id": [{"machine_id": "Machine A", "lift_ratio": 1.8}],
                "shift_id": [{"shift_id": "night", "lift_ratio": 1.6}],
                "operator_id": [{"operator_id": "OP-17", "lift_ratio": 1.2}],
                "defect_type": [{"defect_type": "crack", "lift_ratio": 1.4}],
            },
            "anomalies": {
                "zscore_threshold": 2.0,
                "spikes": [{"hour": "2026-03-07 22:00:00", "defect_count": 78, "zscore": 2.8}],
            },
        }
    }

    print(json.dumps(build_actions(sample_rca), indent=2))