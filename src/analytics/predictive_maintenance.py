from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PredictiveMaintenanceConfig:
    short_window: str = "2h"
    long_window: str = "24h"


class PredictiveMaintenanceAnalyzer:
    def __init__(self, config: PredictiveMaintenanceConfig | None = None):
        self.config = config or PredictiveMaintenanceConfig()

    def analyze(
        self,
        events: pd.DataFrame,
        *,
        anomalies: dict[str, Any] | None = None,
        control_limits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if events.empty or "timestamp" not in events.columns:
            return {
                "risk_score": "low",
                "numeric_risk": 0.0,
                "summary": "Insufficient telemetry for predictive maintenance assessment.",
            }

        working = events.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
        working = working.dropna(subset=["timestamp"]).sort_values("timestamp")
        if working.empty:
            return {
                "risk_score": "low",
                "numeric_risk": 0.0,
                "summary": "Insufficient telemetry for predictive maintenance assessment.",
            }

        indexed = working.set_index("timestamp")
        hourly = indexed.resample("1h").size().rename("defects")
        if hourly.empty:
            return {
                "risk_score": "low",
                "numeric_risk": 0.0,
                "summary": "Insufficient telemetry for predictive maintenance assessment.",
            }

        short_rate = float(hourly.tail(2).mean() or 0.0)
        long_rate = float(hourly.tail(24).mean() or short_rate or 0.0)
        defect_trend = float(np.clip((short_rate - long_rate) / max(long_rate, 1.0), -1.0, 1.0))

        anomaly_frequency = float(len((anomalies or {}).get("spikes", []) or []))
        anomaly_signal = float(np.clip(anomaly_frequency / 5.0, 0.0, 1.0))

        stress_signal = 0.0
        metrics = (control_limits or {}).get("metrics", {}) or {}
        for payload in metrics.values():
            stress_signal += float(payload.get("defect_rate_outside_warning_limits", 0.0) or 0.0)
            stress_signal += 0.05 * float(payload.get("trend_violations", 0) or 0.0)
        if metrics:
            stress_signal = float(np.clip(stress_signal / max(len(metrics), 1), 0.0, 1.0))

        numeric_risk = float(
            np.clip((0.45 * max(defect_trend, 0.0)) + (0.25 * anomaly_signal) + (0.30 * stress_signal), 0.0, 1.0)
        )
        if numeric_risk >= 0.67:
            risk = "high"
        elif numeric_risk >= 0.34:
            risk = "medium"
        else:
            risk = "low"

        machine = "Machine_1"
        if "machine_id" in working.columns and not working["machine_id"].dropna().empty:
            machine = str(working["machine_id"].dropna().astype(str).mode().iloc[0])

        return {
            "risk_score": risk,
            "numeric_risk": round(numeric_risk, 4),
            "drivers": {
                "rolling_defect_rate": round(short_rate, 4),
                "baseline_defect_rate": round(long_rate, 4),
                "anomaly_frequency": int(anomaly_frequency),
                "equipment_stress": round(stress_signal, 4),
            },
            "summary": f"{machine} shows {risk} failure risk in next 2 hours.",
        }


def analyze_predictive_maintenance(
    events: pd.DataFrame,
    *,
    anomalies: dict[str, Any] | None = None,
    control_limits: dict[str, Any] | None = None,
    config: PredictiveMaintenanceConfig | None = None,
) -> dict[str, Any]:
    return PredictiveMaintenanceAnalyzer(config=config).analyze(
        events,
        anomalies=anomalies,
        control_limits=control_limits,
    )