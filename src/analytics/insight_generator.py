from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Insight:
    message: str
    severity: float
    category: str
    confidence: float


def _safe_top(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    return items[0] if items else None


def _select_scope(root: dict[str, Any]) -> dict[str, Any]:
    """Support both single-window RCA and multi-window RCA output."""
    if "trends" in root and "correlations" in root:
        return root
    for key in ["last_hour", "last_24h", "all_time", "last_shift"]:
        if key in root and isinstance(root[key], dict):
            return root[key]
    return root


def _confidence_tag(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def _base_confidence(scope: dict[str, Any]) -> float:
    confidence = scope.get("confidence", {}) or {}
    return float(confidence.get("overall_confidence", 0.5) or 0.5)


def _trend_insight(root: dict[str, Any]) -> Insight | None:
    trends = root.get("trends", {})
    per_hour = trends.get("defects_per_hour", []) or []
    rolling = trends.get("rolling_mean", []) or []
    if len(per_hour) < 2 or len(rolling) < 1:
        return None

    latest = per_hour[-1]
    latest_roll = rolling[-1]
    count = float(latest.get("defect_count", 0) or 0)
    baseline = float(latest_roll.get("rolling_mean", 0) or 0)
    if baseline <= 0:
        return None

    delta_pct = ((count - baseline) / baseline) * 100.0
    hour = latest.get("hour", "latest hour")

    if delta_pct > 20:
        return Insight(
            message=f"Defect volume increased by {delta_pct:.1f}% at {hour} versus recent trend.",
            severity=min(100.0, abs(delta_pct)),
            category="trend",
            confidence=0.75,
        )
    if delta_pct < -20:
        return Insight(
            message=f"Defect volume dropped by {abs(delta_pct):.1f}% at {hour}, indicating improved stability.",
            severity=min(60.0, abs(delta_pct)),
            category="trend",
            confidence=0.7,
        )
    return Insight(
        message=f"Defect volume at {hour} is broadly aligned with recent trend.",
        severity=15.0,
        category="trend",
        confidence=0.55,
    )


def _correlation_insights(root: dict[str, Any]) -> list[Insight]:
    correlations = root.get("correlations", {})
    out: list[Insight] = []

    mappings = [
        ("machine_id", "machine"),
        ("shift_id", "shift"),
        ("operator_id", "operator"),
        ("defect_type", "defect type"),
    ]

    for key, label in mappings:
        top = _safe_top(correlations.get(key, []) or [])
        if not top:
            continue
        contributor = top.get(key, "unknown")
        defect_rate = float(top.get("defect_rate", 0.0) or 0.0)
        lift_ratio = float(top.get("lift_ratio", 1.0) or 1.0)
        avg_conf = float(top.get("avg_confidence", 0.0) or 0.0)
        avg_sev = float(top.get("avg_severity", 0.0) or 0.0)

        severity = min(100.0, (defect_rate * 70.0) + (avg_conf * 20.0) + (avg_sev * 5.0))
        conf = max(0.3, min(0.95, 0.35 + (defect_rate * 0.35) + (max(0.0, lift_ratio - 1.0) * 0.2)))
        out.append(
            Insight(
                message=(
                    f"Highest {label} contributor is '{contributor}' with defect rate "
                    f"{defect_rate:.1%} and lift {lift_ratio:.2f} (avg confidence {avg_conf:.2f}, avg severity {avg_sev:.2f})."
                ),
                severity=severity,
                category="correlation",
                confidence=conf,
            )
        )

    return out


def _anomaly_insights(root: dict[str, Any]) -> list[Insight]:
    anomalies = root.get("anomalies", {})
    spikes = anomalies.get("spikes", []) or []
    if not spikes:
        return [
            Insight(
                message="No significant time-based defect spikes detected in the analyzed window.",
                severity=10.0,
                category="anomaly",
                confidence=0.5,
            )
        ]

    out: list[Insight] = []
    for spike in spikes[:5]:
        hour = spike.get("hour", "unknown time")
        count = float(spike.get("defect_count", 0) or 0)
        zscore = float(spike.get("zscore", 0) or 0)
        out.append(
            Insight(
                message=f"Anomalous spike at {hour}: {count:.0f} defects (z-score {zscore:.2f}).",
                severity=min(100.0, 40.0 + zscore * 20.0),
                category="anomaly",
                confidence=min(0.98, 0.45 + (zscore / 4.0)),
            )
        )
    return out


def _cluster_insight(root: dict[str, Any]) -> Insight | None:
    clusters = root.get("clusters", {})
    note = clusters.get("note")
    if note:
        return Insight(
            message=f"Clustering note: {note}",
            severity=20.0,
            category="cluster",
            confidence=0.45,
        )

    counts = clusters.get("counts", []) or []
    if not counts:
        return None

    dominant = max(counts, key=lambda x: float(x.get("count", 0) or 0))
    cid = dominant.get("cluster", "unknown")
    ccount = int(dominant.get("count", 0) or 0)

    return Insight(
        message=f"Cluster {cid} is dominant with {ccount} events, indicating a recurring defect profile.",
        severity=min(85.0, 30.0 + ccount / 10.0),
        category="cluster",
        confidence=min(0.9, 0.5 + ccount / 200.0),
    )


def _combined_signal_insight(root: dict[str, Any]) -> Insight | None:
    """Create a single narrative from multiple RCA signals."""
    trends = root.get("trends", {})
    per_hour = trends.get("defects_per_hour", []) or []
    rolling = trends.get("rolling_mean", []) or []
    if not per_hour or not rolling:
        return None

    latest = per_hour[-1]
    latest_roll = rolling[-1]
    latest_count = float(latest.get("defect_count", 0) or 0)
    baseline = float(latest_roll.get("rolling_mean", 0) or 0)
    if baseline <= 0:
        return None

    delta_pct = ((latest_count - baseline) / baseline) * 100.0
    if abs(delta_pct) < 15:
        return None

    correlations = root.get("correlations", {})
    top_machine = _safe_top(correlations.get("machine_id", []) or [])
    top_shift = _safe_top(correlations.get("shift_id", []) or [])
    top_defect = _safe_top(correlations.get("defect_type", []) or [])
    anomalies = root.get("anomalies", {})
    spikes = anomalies.get("spikes", []) or []

    machine = (top_machine or {}).get("machine_id", "unknown machine")
    shift = (top_shift or {}).get("shift_id", "unknown shift")
    defect_type = (top_defect or {}).get("defect_type", "defects")
    direction = "increased" if delta_pct > 0 else "decreased"

    if spikes:
        narrative = (
            f"{defect_type.capitalize()} {direction} during {shift} shift on {machine}, "
            f"with concurrent anomaly spikes suggesting a process transition risk."
        )
    else:
        narrative = (
            f"{defect_type.capitalize()} {direction} during {shift} shift on {machine}, "
            f"indicating a repeatable operating-condition pattern."
        )

    severity = min(100.0, 45.0 + abs(delta_pct) * 0.6)
    confidence = min(0.96, 0.55 + (abs(delta_pct) / 200.0) + (0.1 if spikes else 0.0))
    return Insight(message=narrative, severity=severity, category="combined", confidence=confidence)


def generate_insights(root_cause_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Template-based narrative generator with confidence tags, ranked by severity."""
    scope = _select_scope(root_cause_output)
    base_conf = _base_confidence(scope)
    insights: list[Insight] = []

    combined = _combined_signal_insight(scope)
    if combined:
        insights.append(combined)

    t = _trend_insight(scope)
    if t:
        insights.append(t)

    insights.extend(_correlation_insights(scope))
    insights.extend(_anomaly_insights(scope))

    c = _cluster_insight(scope)
    if c:
        insights.append(c)

    insights.sort(key=lambda x: x.severity, reverse=True)
    out: list[dict[str, Any]] = []
    for i in insights:
        final_conf = max(0.05, min(0.99, (0.6 * i.confidence) + (0.4 * base_conf)))
        out.append(
            {
                "insight": i.message,
                "category": i.category,
                "severity": round(i.severity, 2),
                "confidence": round(final_conf, 4),
                "confidence_tag": _confidence_tag(final_conf),
            }
        )
    return out


if __name__ == "__main__":
    # Standalone test example
    example_input = {
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
            "machine_id": [{"machine_id": "Machine A", "defect_rate": 0.46, "avg_confidence": 0.81, "avg_severity": 2.7}],
            "shift_id": [{"shift_id": "night", "defect_rate": 0.51, "avg_confidence": 0.77, "avg_severity": 2.5}],
            "operator_id": [{"operator_id": "OP-17", "defect_rate": 0.31, "avg_confidence": 0.69, "avg_severity": 2.2}],
            "defect_type": [{"defect_type": "crack", "defect_rate": 0.58, "avg_confidence": 0.83, "avg_severity": 2.9}],
        },
        "anomalies": {
            "spikes": [{"hour": "2026-03-07 22:00:00", "defect_count": 78, "zscore": 2.8}]
        },
        "clusters": {
            "k": 3,
            "counts": [{"cluster": 0, "count": 89}, {"cluster": 1, "count": 31}, {"cluster": 2, "count": 14}],
            "centroids": [],
        },
        "confidence": {
            "overall_confidence": 0.78,
            "insights": [
                {"insight": "anomaly_signal", "confidence": 0.82},
                {"insight": "correlation_signal", "confidence": 0.71},
                {"insight": "consistency_over_time", "confidence": 0.79},
            ],
        },
    }

    generated = generate_insights(example_input)
    print(json.dumps(generated, indent=2))