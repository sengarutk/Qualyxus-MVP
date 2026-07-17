from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analytics.causal_engine import score_causal_factors
from src.analytics.control_limits import analyze_control_limits
from src.analytics.predictive_maintenance import analyze_predictive_maintenance
from src.analytics.temporal_rca import analyze_temporal_rca


DEFAULT_DB_PATH = "data/events/events.db"


@dataclass
class RootCauseConfig:
    sqlite_path: str = DEFAULT_DB_PATH
    rolling_window: int = 5
    anomaly_z_threshold: float = 2.0
    max_contributors_per_group: int = 5
    kmeans_clusters: int = 4
    production_volume: float | None = None
    scrap_rate: float | None = None
    cycle_time: float | None = None


class RootCauseIntelligence:
    def __init__(self, config: RootCauseConfig | None = None):
        self.config = config or RootCauseConfig()

    def load_events(self) -> pd.DataFrame:
        """Load defect events from SQLite and normalize timestamp/features."""
        db_path = Path(self.config.sqlite_path)
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found: {db_path}")

        with sqlite3.connect(str(db_path)) as conn:
            cols = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(defect_events)").fetchall()
            }
            select_columns = [
                "event_id",
                "timestamp",
                "source",
                "defect_type",
                "confidence",
                "severity",
                "batch_id",
                "operator_id",
                "machine_id",
                "shift_id",
                "line_id",
                "station_id",
                "temperature",
                "vibration",
                "speed",
                "lot_id",
            ]
            query = "SELECT {} FROM defect_events".format(
                ",\n                ".join(
                    [
                        col if col in cols else f"NULL AS {col}"
                        for col in select_columns
                    ]
                )
            )
            events = pd.read_sql_query(query, conn)

        if events.empty:
            return events

        events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
        events = events.dropna(subset=["timestamp"]).copy()

        events["confidence"] = pd.to_numeric(events["confidence"], errors="coerce")
        events["severity"] = pd.to_numeric(events["severity"], errors="coerce")

        events["defect_type"] = events["defect_type"].fillna("unknown").astype(str)
        for col in ["machine_id", "shift_id", "operator_id", "source"]:
            events[col] = events[col].fillna("unknown").astype(str)

        events["hour"] = events["timestamp"].dt.hour
        return events

    def analyze_trends(self, events: pd.DataFrame) -> dict[str, Any]:
        if events.empty:
            return {"defects_per_hour": [], "rolling_mean": []}

        per_hour = (
            events.set_index("timestamp")
            .resample("h")
            .size()
            .rename("defect_count")
            .to_frame()
            .reset_index()
        )
        per_hour["hour"] = per_hour["timestamp"].dt.strftime("%Y-%m-%d %H:00:00")
        per_hour["rolling_mean"] = (
            per_hour["defect_count"].rolling(window=self.config.rolling_window, min_periods=1).mean()
        )

        return {
            "defects_per_hour": per_hour[["hour", "defect_count"]].to_dict(orient="records"),
            "rolling_mean": per_hour[["hour", "rolling_mean"]].to_dict(orient="records"),
        }

    @staticmethod
    def detect_systemic_clusters(events: pd.DataFrame) -> list[dict[str, Any]]:
        if events.empty:
            return []

        working = events.copy()
        working["severity"] = pd.to_numeric(working["severity"], errors="coerce").fillna(0)
        working = working[working["severity"] >= 2].sort_values("timestamp")
        if working.empty:
            return []

        clusters: list[dict[str, Any]] = []
        for idx in range(len(working)):
            start_ts = working.iloc[idx]["timestamp"]
            end_ts = start_ts + pd.Timedelta(minutes=10)
            window = working[(working["timestamp"] >= start_ts) & (working["timestamp"] <= end_ts)]
            if len(window) <= 4:
                continue
            clusters.append(
                {
                    "cluster_id": f"cluster:{start_ts.isoformat()}",
                    "window_start": start_ts.isoformat(),
                    "window_end": end_ts.isoformat(),
                    "event_ids": window["event_id"].astype(str).tolist(),
                    "camera_sources": sorted(window["source"].astype(str).unique().tolist()),
                    "event_count": int(len(window)),
                    "telemetry_snapshot": {
                        "avg_speed": round(float(pd.to_numeric(window.get("speed"), errors="coerce").mean() or 0.0), 4),
                        "avg_temperature": round(float(pd.to_numeric(window.get("temperature"), errors="coerce").mean() or 0.0), 4),
                        "avg_vibration": round(float(pd.to_numeric(window.get("vibration"), errors="coerce").mean() or 0.0), 4),
                    },
                    "events": window[[
                        "event_id", "timestamp", "source", "defect_type", "severity", "confidence", "line_id", "station_id"
                    ]].to_dict(orient="records"),
                }
            )
            break
        return clusters[:1]

    @staticmethod
    def _build_baseline_rates(events: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Baseline frequencies by dimension for lift-based correlation ranking."""
        if events.empty:
            return {"machine_id": {}, "shift_id": {}, "operator_id": {}, "defect_type": {}}

        total = float(len(events))
        out: dict[str, dict[str, float]] = {}
        for col in ["machine_id", "shift_id", "operator_id", "defect_type"]:
            series = events[col].fillna("unknown").astype(str)
            rates = (series.value_counts(dropna=False) / total).to_dict()
            out[col] = {str(k): float(v) for k, v in rates.items()}
        return out

    def analyze_correlations(
        self,
        events: pd.DataFrame,
        baseline_rates: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, Any]:
        if events.empty:
            return {
                "machine_id": [],
                "shift_id": [],
                "operator_id": [],
                "defect_type": [],
            }

        total_events = float(len(events))
        baseline_rates = baseline_rates or {}
        eps = 1e-9

        def grouped_top(column: str) -> list[dict[str, Any]]:
            grouped = (
                events.groupby(column, dropna=False)
                .agg(
                    defect_count=("event_id", "count"),
                    avg_confidence=("confidence", "mean"),
                    avg_severity=("severity", "mean"),
                )
                .reset_index()
            )
            grouped["defect_rate"] = grouped["defect_count"] / total_events

            baseline_map = baseline_rates.get(column, {})
            grouped["baseline_rate"] = grouped[column].astype(str).map(baseline_map).fillna(0.0)
            grouped["lift_ratio"] = grouped["defect_rate"] / (grouped["baseline_rate"] + eps)

            # Rank by normalized impact (lift), then support (count).
            grouped = grouped.sort_values(["lift_ratio", "defect_count"], ascending=False)
            top = grouped.head(self.config.max_contributors_per_group)
            return top.to_dict(orient="records")

        return {
            "machine_id": grouped_top("machine_id"),
            "shift_id": grouped_top("shift_id"),
            "operator_id": grouped_top("operator_id"),
            "defect_type": grouped_top("defect_type"),
            "temperature": grouped_top("temperature"),
            "vibration": grouped_top("vibration"),
            "speed": grouped_top("speed"),
            "lot_id": grouped_top("lot_id"),
        }

    def detect_time_anomalies(self, events: pd.DataFrame) -> dict[str, Any]:
        if events.empty:
            return {"spikes": [], "zscore_threshold": self.config.anomaly_z_threshold}

        counts = (
            events.set_index("timestamp")
            .resample("h")
            .size()
            .rename("defect_count")
            .to_frame()
            .reset_index()
        )
        if counts.empty:
            return {"spikes": [], "zscore_threshold": self.config.anomaly_z_threshold}

        rolling_mean = counts["defect_count"].rolling(window=self.config.rolling_window, min_periods=2).mean()
        rolling_std = counts["defect_count"].rolling(window=self.config.rolling_window, min_periods=2).std(ddof=0)
        zscore = (counts["defect_count"] - rolling_mean) / rolling_std.replace(0, np.nan)

        counts["zscore"] = zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        spikes = counts[counts["zscore"] >= self.config.anomaly_z_threshold].copy()
        spikes["hour"] = spikes["timestamp"].dt.strftime("%Y-%m-%d %H:00:00")

        return {
            "zscore_threshold": self.config.anomaly_z_threshold,
            "spikes": spikes[["hour", "defect_count", "zscore"]].to_dict(orient="records"),
        }

    def cluster_events(self, events: pd.DataFrame) -> dict[str, Any]:
        if events.empty:
            return {"k": 0, "counts": [], "centroids": [], "note": "No events available"}

        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:  # pragma: no cover - dependency dependent
            return {
                "k": 0,
                "counts": [],
                "centroids": [],
                "note": f"sklearn unavailable: {exc}",
            }

        working = events[["confidence", "hour", "defect_type"]].copy()
        working["confidence"] = pd.to_numeric(working["confidence"], errors="coerce").fillna(0.0)
        working["hour"] = pd.to_numeric(working["hour"], errors="coerce").fillna(0.0)
        working["defect_type_encoded"] = pd.factorize(working["defect_type"].fillna("unknown"))[0].astype(float)

        features = working[["confidence", "hour", "defect_type_encoded"]].to_numpy(dtype=float)
        if len(features) < 2:
            return {
                "k": 1,
                "counts": [{"cluster": 0, "count": int(len(features))}],
                "centroids": [],
                "note": "Insufficient samples for meaningful clustering",
            }

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)

        k = max(2, min(self.config.kmeans_clusters, len(features)))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)

        counts = pd.Series(labels, name="cluster").value_counts().sort_index()
        centers = scaler.inverse_transform(model.cluster_centers_)

        centroid_rows: list[dict[str, Any]] = []
        for cluster_id, center in enumerate(centers):
            centroid_rows.append(
                {
                    "cluster": int(cluster_id),
                    "confidence": float(center[0]),
                    "hour": float(center[1]),
                    "defect_type_encoded": float(center[2]),
                }
            )

        return {
            "k": int(k),
            "counts": [{"cluster": int(idx), "count": int(val)} for idx, val in counts.items()],
            "centroids": centroid_rows,
        }

    @staticmethod
    def _current_shift_start(ts: pd.Timestamp) -> pd.Timestamp:
        """
        Shift model (UTC, 8h blocks):
        - A: 06:00-14:00
        - B: 14:00-22:00
        - C: 22:00-06:00
        Returns the start timestamp for the shift containing `ts`.
        """
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        h = ts.hour

        if 6 <= h < 14:
            return ts.replace(hour=6, minute=0, second=0, microsecond=0)
        if 14 <= h < 22:
            return ts.replace(hour=14, minute=0, second=0, microsecond=0)

        # Night shift C spans 22:00 -> 06:00 (cross-day)
        if h >= 22:
            return ts.replace(hour=22, minute=0, second=0, microsecond=0)
        # 00:00-05:59 belongs to previous day's 22:00 shift start
        prev_day = (ts - pd.Timedelta(days=1)).replace(hour=22, minute=0, second=0, microsecond=0)
        return prev_day

    def _analyze_window(
        self,
        events: pd.DataFrame,
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
        baseline_rates: dict[str, dict[str, float]],
        window_name: str,
        kpi_inputs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        window_events = events[(events["timestamp"] >= start_ts) & (events["timestamp"] < end_ts)].copy()
        trends = self.analyze_trends(window_events)
        correlations = self.analyze_correlations(window_events, baseline_rates=baseline_rates)
        anomalies = self.detect_time_anomalies(window_events)
        clusters = self.cluster_events(window_events)
        temporal_rca = analyze_temporal_rca(window_events)
        control_limits = analyze_control_limits(window_events)
        causal_scoring = score_causal_factors(
            correlations=correlations,
            temporal_rca=temporal_rca,
            control_limits=control_limits,
        )
        predictive_maintenance = analyze_predictive_maintenance(
            window_events,
            anomalies=anomalies,
            control_limits=control_limits,
        )
        classification_6m = self._classify_6m_findings(trends=trends, correlations=correlations, anomalies=anomalies)
        kpi_context = self._resolve_window_kpis(window_name=window_name, kpi_inputs=kpi_inputs)

        return {
            "window_start": start_ts.isoformat(),
            "window_end": end_ts.isoformat(),
            "event_count": int(len(window_events)),
            "trends": trends,
            "correlations": correlations,
            "anomalies": anomalies,
            "clusters": clusters,
            "temporal_rca": temporal_rca,
            "control_limits": control_limits,
            "causal_scoring": causal_scoring,
            "predictive_maintenance": predictive_maintenance,
            "confidence": self._confidence_bundle(
                trends,
                correlations,
                anomalies,
                kpi_context,
                temporal_rca=temporal_rca,
                causal_scoring=causal_scoring,
            ),
            "kpi_context": kpi_context,
            "classification_6m": classification_6m,
            "explanations": self._build_explanations(
                trends=trends,
                correlations=correlations,
                anomalies=anomalies,
                temporal_rca=temporal_rca,
                window_start=start_ts.isoformat(),
                window_end=end_ts.isoformat(),
            ),
        }

    def _resolve_window_kpis(self, *, window_name: str, kpi_inputs: dict[str, Any] | None) -> dict[str, Any] | None:
        """Resolve optional production KPI inputs for a given RCA window."""
        kpi_inputs = kpi_inputs or {}

        def _pick(name: str, default_val: float | None) -> float | None:
            val = kpi_inputs.get(name, default_val)
            if isinstance(val, dict):
                val = val.get(window_name, val.get("default"))
            if val is None:
                return None
            try:
                return float(val)
            except Exception:
                return None

        production_volume = _pick("production_volume", self.config.production_volume)
        scrap_rate = _pick("scrap_rate", self.config.scrap_rate)
        cycle_time = _pick("cycle_time", self.config.cycle_time)

        if production_volume is None and scrap_rate is None and cycle_time is None:
            return None

        def _clip01(x: float) -> float:
            return float(np.clip(x, 0.0, 1.0))

        volume_norm = _clip01((production_volume or 0.0) / 1000.0)
        scrap_raw = float(scrap_rate or 0.0)
        scrap_ratio = (scrap_raw / 100.0) if scrap_raw > 1.0 else scrap_raw
        scrap_norm = _clip01(scrap_ratio)
        cycle_norm = _clip01((cycle_time or 0.0) / 120.0)

        kpi_signal = _clip01((0.20 * volume_norm) + (0.50 * scrap_norm) + (0.30 * cycle_norm))
        return {
            "production_volume": production_volume,
            "scrap_rate": scrap_rate,
            "cycle_time": cycle_time,
            "normalized": {
                "volume_norm": round(volume_norm, 4),
                "scrap_norm": round(scrap_norm, 4),
                "cycle_norm": round(cycle_norm, 4),
            },
            "kpi_signal": round(kpi_signal, 4),
        }

    @staticmethod
    def _classify_6m_findings(
        *,
        trends: dict[str, Any],
        correlations: dict[str, Any],
        anomalies: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Deterministic 6M classification for RCA findings."""
        findings: list[dict[str, Any]] = []

        def _top_lift(dim: str) -> tuple[str, float]:
            rows = correlations.get(dim, []) or []
            if not rows:
                return "unknown", 1.0
            top = rows[0]
            return str(top.get(dim, "unknown")), float(top.get("lift_ratio", 1.0) or 1.0)

        operator, operator_lift = _top_lift("operator_id")
        if operator_lift > 1.0:
            findings.append(
                {
                    "classification_6m": "Man",
                    "basis": "operator correlation",
                    "evidence": {"operator_id": operator, "lift_ratio": round(operator_lift, 4)},
                }
            )

        machine, machine_lift = _top_lift("machine_id")
        if machine_lift > 1.0:
            findings.append(
                {
                    "classification_6m": "Machine",
                    "basis": "machine correlation",
                    "evidence": {"machine_id": machine, "lift_ratio": round(machine_lift, 4)},
                }
            )

        shift, shift_lift = _top_lift("shift_id")
        if shift_lift > 1.0:
            findings.append(
                {
                    "classification_6m": "Environment",
                    "basis": "shift/environment correlation",
                    "evidence": {"shift_id": shift, "lift_ratio": round(shift_lift, 4)},
                }
            )

        defect_type, defect_lift = _top_lift("defect_type")
        if defect_lift > 1.0:
            findings.append(
                {
                    "classification_6m": "Material",
                    "basis": "defect pattern correlation",
                    "evidence": {"defect_type": defect_type, "lift_ratio": round(defect_lift, 4)},
                }
            )

        per_hour = trends.get("defects_per_hour", []) or []
        rolling = trends.get("rolling_mean", []) or []
        if len(per_hour) >= 2 and rolling:
            latest_count = float(per_hour[-1].get("defect_count", 0.0) or 0.0)
            latest_roll = float(rolling[-1].get("rolling_mean", 0.0) or 0.0)
            if latest_roll > 0:
                findings.append(
                    {
                        "classification_6m": "Method",
                        "basis": "time/process sequence trend",
                        "evidence": {
                            "latest_defect_count": latest_count,
                            "rolling_mean": round(latest_roll, 4),
                            "delta_ratio": round((latest_count - latest_roll) / latest_roll, 4),
                        },
                    }
                )

        spikes = anomalies.get("spikes", []) or []
        if spikes:
            max_z = max(float(s.get("zscore", 0.0) or 0.0) for s in spikes)
            findings.append(
                {
                    "classification_6m": "Measurement",
                    "basis": "sensor/process deviation anomalies",
                    "evidence": {
                        "spike_count": len(spikes),
                        "max_zscore": round(max_z, 4),
                        "zscore_threshold": float(anomalies.get("zscore_threshold", 2.0) or 2.0),
                    },
                }
            )

        return findings

    @staticmethod
    def _build_explanations(
        *,
        trends: dict[str, Any],
        correlations: dict[str, Any],
        anomalies: dict[str, Any],
        temporal_rca: dict[str, Any] | None = None,
        window_start: str,
        window_end: str,
    ) -> list[dict[str, Any]]:
        """Structured explainability payload for audit/UI narration."""
        explanations: list[dict[str, Any]] = []

        spikes = anomalies.get("spikes", []) or []
        top_machine = (correlations.get("machine_id", []) or [{}])[0] if correlations.get("machine_id") else {}
        top_shift = (correlations.get("shift_id", []) or [{}])[0] if correlations.get("shift_id") else {}

        if spikes:
            max_z = max(float(s.get("zscore", 0.0) or 0.0) for s in spikes)
            machine = str(top_machine.get("machine_id", "unknown"))
            machine_lift = float(top_machine.get("lift_ratio", 1.0) or 1.0)
            explanations.append(
                {
                    "insight": (
                        f"Spike detected due to {len(spikes)} consecutive anomaly events "
                        f"and {machine_lift:.2f}x lift on {machine}."
                    ),
                    "classification_6m": "Measurement" if machine_lift <= 1.0 else "Machine",
                    "contributing_signals": ["anomaly_strength", "correlation_lift"],
                    "data_window_used": {"start": window_start, "end": window_end},
                    "evidence_summary": {
                        "anomaly_event_count": int(len(spikes)),
                        "max_zscore": round(max_z, 4),
                        "top_machine": machine,
                        "top_machine_lift_ratio": round(machine_lift, 4),
                    },
                }
            )

        # Trend explainability
        per_hour = trends.get("defects_per_hour", []) or []
        rolling = trends.get("rolling_mean", []) or []
        if per_hour and rolling:
            latest_count = float(per_hour[-1].get("defect_count", 0.0) or 0.0)
            latest_roll = float(rolling[-1].get("rolling_mean", 0.0) or 0.0)
            if latest_roll > 0:
                delta_ratio = (latest_count - latest_roll) / latest_roll
                shift = str(top_shift.get("shift_id", "unknown"))
                shift_lift = float(top_shift.get("lift_ratio", 1.0) or 1.0)
                explanations.append(
                    {
                        "insight": (
                            f"Frequency trend changed by {delta_ratio * 100:.1f}% with {shift_lift:.2f}x lift on shift '{shift}'."
                        ),
                        "classification_6m": "Environment" if shift_lift > 1.0 else "Method",
                        "contributing_signals": ["frequency_trend", "correlation_lift"],
                        "data_window_used": {"start": window_start, "end": window_end},
                        "evidence_summary": {
                            "latest_defect_count": latest_count,
                            "rolling_mean": round(latest_roll, 4),
                            "delta_ratio": round(delta_ratio, 4),
                            "top_shift": shift,
                            "top_shift_lift_ratio": round(shift_lift, 4),
                        },
                    }
                )

        temporal_insights = (temporal_rca or {}).get("insights", []) or []
        for item in temporal_insights[:2]:
            explanations.append(
                {
                    "insight": str(item.get("statement", "Temporal telemetry pattern detected.")),
                    "classification_6m": "Method",
                    "contributing_signals": ["temporal_alignment", str(item.get("signal", "telemetry"))],
                    "data_window_used": {"start": window_start, "end": window_end},
                    "evidence_summary": {
                        "signal": item.get("signal"),
                        "lag_seconds": item.get("lag_seconds"),
                        "correlation": item.get("correlation"),
                        "drift_correlation": item.get("drift_correlation"),
                    },
                }
            )

        return explanations

    @staticmethod
    def _anomaly_strength(anomalies: dict[str, Any]) -> float:
        spikes = anomalies.get("spikes", []) or []
        threshold = float(anomalies.get("zscore_threshold", 2.0) or 2.0)
        if not spikes:
            return 0.1

        max_z = max(float(s.get("zscore", 0.0) or 0.0) for s in spikes)
        # 1.0 reached around ~2x threshold.
        return float(np.clip(max_z / max(threshold * 2.0, 1e-9), 0.0, 1.0))

    @staticmethod
    def _correlation_strength(correlations: dict[str, Any]) -> float:
        lifts: list[float] = []
        supports: list[float] = []

        for dim in ["machine_id", "shift_id", "operator_id", "defect_type"]:
            rows = correlations.get(dim, []) or []
            if not rows:
                continue
            top = rows[0]
            lift = float(top.get("lift_ratio", 1.0) or 1.0)
            support = float(top.get("defect_rate", 0.0) or 0.0)
            lifts.append(lift)
            supports.append(support)

        if not lifts:
            return 0.1

        # Normalize lift around neutral point=1.0
        avg_lift = float(np.mean(lifts))
        lift_signal = np.clip((avg_lift - 1.0) / 2.0, 0.0, 1.0)
        support_signal = float(np.clip(np.mean(supports), 0.0, 1.0))
        return float(np.clip(0.7 * lift_signal + 0.3 * support_signal, 0.0, 1.0))

    @staticmethod
    def _consistency_strength(trends: dict[str, Any]) -> float:
        points = trends.get("defects_per_hour", []) or []
        if len(points) < 2:
            return 0.5

        values = np.array([float(p.get("defect_count", 0.0) or 0.0) for p in points], dtype=float)
        mean_v = float(values.mean())
        if mean_v <= 1e-9:
            return 0.5

        cv = float(values.std(ddof=0) / mean_v)  # coefficient of variation
        # Lower variability => higher consistency confidence.
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _confidence_bundle(
        self,
        trends: dict[str, Any],
        correlations: dict[str, Any],
        anomalies: dict[str, Any],
        kpi_context: dict[str, Any] | None = None,
        temporal_rca: dict[str, Any] | None = None,
        causal_scoring: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        anomaly_strength = self._anomaly_strength(anomalies)
        correlation_strength = self._correlation_strength(correlations)
        consistency_strength = self._consistency_strength(trends)
        kpi_signal = float((kpi_context or {}).get("kpi_signal", 0.0) or 0.0)
        temporal_strength = 0.0
        temporal_insights = (temporal_rca or {}).get("insights", []) or []
        if temporal_insights:
            temporal_strength = float(
                np.clip(
                    np.mean(
                        [
                            max(
                                abs(float(item.get("correlation", 0.0) or 0.0)),
                                abs(float(item.get("drift_correlation", 0.0) or 0.0)),
                            )
                            for item in temporal_insights[:3]
                        ]
                    ),
                    0.0,
                    1.0,
                )
            )
        causal_factor_confidence = (causal_scoring or {}).get("factor_confidence", {}) or {}
        causal_strength = float(np.mean(list(causal_factor_confidence.values()))) if causal_factor_confidence else 0.0

        # Weighted confidence combining requested signals.
        if kpi_context:
            overall = float(
                np.clip(
                    (0.24 * anomaly_strength)
                    + (0.24 * correlation_strength)
                    + (0.20 * consistency_strength)
                    + (0.12 * kpi_signal)
                    + (0.10 * temporal_strength)
                    + (0.10 * causal_strength),
                    0.0,
                    1.0,
                )
            )
        else:
            overall = float(
                np.clip(
                    (0.28 * anomaly_strength)
                    + (0.28 * correlation_strength)
                    + (0.24 * consistency_strength)
                    + (0.10 * temporal_strength)
                    + (0.10 * causal_strength),
                    0.0,
                    1.0,
                )
            )

        insights = [
            {
                "insight": "anomaly_signal",
                "confidence": round(anomaly_strength, 4),
                "description": "Confidence derived from spike z-score intensity.",
            },
            {
                "insight": "correlation_signal",
                "confidence": round(correlation_strength, 4),
                "description": "Confidence derived from lift-normalized correlation strength.",
            },
            {
                "insight": "consistency_over_time",
                "confidence": round(consistency_strength, 4),
                "description": "Confidence derived from hourly trend stability.",
            },
        ]

        if kpi_context:
            insights.append(
                {
                    "insight": "production_kpi_signal",
                    "confidence": round(kpi_signal, 4),
                    "description": "Confidence contribution from production volume, scrap rate, and cycle time.",
                }
            )

        if temporal_insights:
            insights.append(
                {
                    "insight": "temporal_rca_signal",
                    "confidence": round(temporal_strength, 4),
                    "description": "Confidence contribution from lag-based telemetry-to-defect alignment.",
                }
            )

        if causal_factor_confidence:
            insights.append(
                {
                    "insight": "causal_inference_signal",
                    "confidence": round(causal_strength, 4),
                    "description": "Confidence contribution from additive causal factor scoring.",
                }
            )

        return {
            "overall_confidence": round(overall, 4),
            "insights": insights,
            "factor_confidence": {k: round(float(v), 4) for k, v in causal_factor_confidence.items()},
        }

    def run(self, kpi_inputs: dict[str, Any] | None = None) -> dict[str, Any]:
        events = self.load_events()
        baseline_rates = self._build_baseline_rates(events)

        # Legacy all-time RCA output retained for compatibility.
        all_time_trends = self.analyze_trends(events)
        all_time_correlations = self.analyze_correlations(events, baseline_rates=baseline_rates)
        all_time_anomalies = self.detect_time_anomalies(events)
        all_time_clusters = self.cluster_events(events)
        all_time_temporal_rca = analyze_temporal_rca(events)
        all_time_control_limits = analyze_control_limits(events)
        all_time_causal_scoring = score_causal_factors(
            correlations=all_time_correlations,
            temporal_rca=all_time_temporal_rca,
            control_limits=all_time_control_limits,
        )
        all_time_predictive_maintenance = analyze_predictive_maintenance(
            events,
            anomalies=all_time_anomalies,
            control_limits=all_time_control_limits,
        )

        all_time = {
            "trends": all_time_trends,
            "correlations": all_time_correlations,
            "anomalies": all_time_anomalies,
            "clusters": all_time_clusters,
            "temporal_rca": all_time_temporal_rca,
            "control_limits": all_time_control_limits,
            "causal_scoring": all_time_causal_scoring,
            "predictive_maintenance": all_time_predictive_maintenance,
            "confidence": self._confidence_bundle(
                all_time_trends,
                all_time_correlations,
                all_time_anomalies,
                self._resolve_window_kpis(window_name="all_time", kpi_inputs=kpi_inputs),
                temporal_rca=all_time_temporal_rca,
                causal_scoring=all_time_causal_scoring,
            ),
            "kpi_context": self._resolve_window_kpis(window_name="all_time", kpi_inputs=kpi_inputs),
            "classification_6m": self._classify_6m_findings(
                trends=all_time_trends,
                correlations=all_time_correlations,
                anomalies=all_time_anomalies,
            ),
            "explanations": self._build_explanations(
                trends=all_time_trends,
                correlations=all_time_correlations,
                anomalies=all_time_anomalies,
                temporal_rca=all_time_temporal_rca,
                window_start=(
                    events["timestamp"].min().isoformat()
                    if not events.empty
                    else pd.Timestamp.utcnow().isoformat()
                ),
                window_end=(
                    events["timestamp"].max().isoformat()
                    if not events.empty
                    else pd.Timestamp.utcnow().isoformat()
                ),
            ),
            "systemic_clusters": self.detect_systemic_clusters(events),
        }

        if events.empty:
            now = pd.Timestamp.utcnow().tz_localize("UTC") if pd.Timestamp.utcnow().tzinfo is None else pd.Timestamp.utcnow()
            return {
                "last_hour": self._analyze_window(
                    events,
                    now - timedelta(hours=1),
                    now,
                    baseline_rates,
                    window_name="last_hour",
                    kpi_inputs=kpi_inputs,
                ),
                "last_shift": self._analyze_window(
                    events,
                    now - timedelta(hours=8),
                    now,
                    baseline_rates,
                    window_name="last_shift",
                    kpi_inputs=kpi_inputs,
                ),
                "last_24h": self._analyze_window(
                    events,
                    now - timedelta(hours=24),
                    now,
                    baseline_rates,
                    window_name="last_24h",
                    kpi_inputs=kpi_inputs,
                ),
                "all_time": all_time,
            }

        reference_ts = events["timestamp"].max()
        if reference_ts.tzinfo is None:
            reference_ts = reference_ts.tz_localize("UTC")

        last_hour_start = reference_ts - timedelta(hours=1)
        last_24h_start = reference_ts - timedelta(hours=24)

        current_shift_start = self._current_shift_start(reference_ts)
        last_shift_end = current_shift_start
        last_shift_start = last_shift_end - timedelta(hours=8)

        return {
            "last_hour": self._analyze_window(
                events,
                last_hour_start,
                reference_ts,
                baseline_rates,
                window_name="last_hour",
                kpi_inputs=kpi_inputs,
            ),
            "last_shift": self._analyze_window(
                events,
                last_shift_start,
                last_shift_end,
                baseline_rates,
                window_name="last_shift",
                kpi_inputs=kpi_inputs,
            ),
            "last_24h": self._analyze_window(
                events,
                last_24h_start,
                reference_ts,
                baseline_rates,
                window_name="last_24h",
                kpi_inputs=kpi_inputs,
            ),
            "all_time": all_time,
        }


def run_root_cause(
    sqlite_path: str = DEFAULT_DB_PATH,
    kpi_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    engine = RootCauseIntelligence(config=RootCauseConfig(sqlite_path=sqlite_path))
    return engine.run(kpi_inputs=kpi_inputs)


if __name__ == "__main__":
    summary = run_root_cause()
    print(json.dumps(summary, indent=2, default=str))
