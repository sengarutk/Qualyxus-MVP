from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_LAG_WINDOWS_SECONDS = (30, 60, 120, 300)
NUMERIC_SIGNALS = ("temperature", "vibration", "speed")


@dataclass(frozen=True)
class TemporalRCAConfig:
    lag_windows_seconds: tuple[int, ...] = DEFAULT_LAG_WINDOWS_SECONDS
    aggregation_frequency: str = "30s"
    min_points: int = 6
    spike_quantile: float = 0.85


class TemporalRCAAnalyzer:
    """Additive temporal analysis for telemetry conditions preceding defect spikes."""

    def __init__(self, config: TemporalRCAConfig | None = None):
        self.config = config or TemporalRCAConfig()

    def analyze(self, events: pd.DataFrame) -> dict[str, Any]:
        if events.empty or "timestamp" not in events.columns:
            return {
                "lag_windows_seconds": list(self.config.lag_windows_seconds),
                "aggregation_frequency": self.config.aggregation_frequency,
                "signal_summaries": {},
                "insights": [],
            }

        working = events.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
        working = working.dropna(subset=["timestamp"]).sort_values("timestamp")
        if working.empty:
            return {
                "lag_windows_seconds": list(self.config.lag_windows_seconds),
                "aggregation_frequency": self.config.aggregation_frequency,
                "signal_summaries": {},
                "insights": [],
            }

        indexed = working.set_index("timestamp")
        defect_series = indexed.resample(self.config.aggregation_frequency).size().rename("defect_count")
        if defect_series.empty:
            return {
                "lag_windows_seconds": list(self.config.lag_windows_seconds),
                "aggregation_frequency": self.config.aggregation_frequency,
                "signal_summaries": {},
                "insights": [],
            }

        signal_summaries: dict[str, Any] = {}
        insights: list[dict[str, Any]] = []

        for signal in NUMERIC_SIGNALS:
            if signal not in indexed.columns:
                continue
            signal_series = pd.to_numeric(indexed[signal], errors="coerce").dropna()
            if signal_series.empty:
                continue

            resampled_signal = signal_series.resample(self.config.aggregation_frequency).mean()
            aligned = pd.concat([defect_series, resampled_signal.rename(signal)], axis=1).dropna()
            if len(aligned) < self.config.min_points:
                continue

            baseline_mean = float(aligned[signal].mean())
            baseline_std = float(aligned[signal].std(ddof=0) or 0.0)
            if baseline_std > 0:
                aligned[f"{signal}_zscore"] = (aligned[signal] - baseline_mean) / baseline_std
            else:
                aligned[f"{signal}_zscore"] = 0.0
            aligned[f"{signal}_drift"] = aligned[signal].diff().fillna(0.0)

            lag_records: list[dict[str, Any]] = []
            best_record: dict[str, Any] | None = None
            for lag_seconds in self.config.lag_windows_seconds:
                periods = max(1, int(round(lag_seconds / pd.Timedelta(self.config.aggregation_frequency).total_seconds())))
                lagged = aligned[signal].shift(periods)
                lagged_drift = aligned[f"{signal}_drift"].shift(periods)
                pair = pd.concat(
                    [aligned["defect_count"], lagged.rename("lagged_signal"), lagged_drift.rename("lagged_drift")],
                    axis=1,
                ).dropna()
                if len(pair) < self.config.min_points:
                    continue

                corr = float(pair["lagged_signal"].corr(pair["defect_count"]))
                drift_corr = float(pair["lagged_drift"].corr(pair["defect_count"]))
                if np.isnan(corr):
                    corr = 0.0
                if np.isnan(drift_corr):
                    drift_corr = 0.0

                spike_threshold = float(pair["lagged_signal"].quantile(self.config.spike_quantile))
                high_signal = pair[pair["lagged_signal"] >= spike_threshold]
                low_signal = pair[pair["lagged_signal"] < spike_threshold]
                spike_defect_delta = 0.0
                if not high_signal.empty and not low_signal.empty:
                    spike_defect_delta = float(
                        high_signal["defect_count"].mean() - low_signal["defect_count"].mean()
                    )

                record = {
                    "lag_seconds": int(lag_seconds),
                    "correlation": round(corr, 4),
                    "drift_correlation": round(drift_corr, 4),
                    "spike_defect_delta": round(spike_defect_delta, 4),
                    "sample_size": int(len(pair)),
                }
                lag_records.append(record)

                strength = abs(corr) + (0.35 * abs(drift_corr)) + (0.15 * max(spike_defect_delta, 0.0))
                if best_record is None or strength > best_record["_strength"]:
                    best_record = {**record, "_strength": strength}

            if not lag_records:
                continue

            signal_summaries[signal] = {
                "baseline_mean": round(baseline_mean, 4),
                "baseline_std": round(baseline_std, 4),
                "lags": lag_records,
            }

            if best_record is None:
                continue

            relation = "increase" if best_record["correlation"] >= 0 else "decrease"
            phenomenon = "spikes" if best_record["spike_defect_delta"] >= 0 else "drops"
            if abs(best_record["drift_correlation"]) > abs(best_record["correlation"]):
                wording = f"{signal.replace('_', ' ').title()} drift precedes anomaly clusters"
            else:
                wording = f"Defects {relation} {best_record['lag_seconds'] // 60 or '<1'} minutes after {signal} {phenomenon}"

            insights.append(
                {
                    "signal": signal,
                    "lag_seconds": int(best_record["lag_seconds"]),
                    "correlation": round(float(best_record["correlation"]), 4),
                    "drift_correlation": round(float(best_record["drift_correlation"]), 4),
                    "spike_defect_delta": round(float(best_record["spike_defect_delta"]), 4),
                    "statement": wording.replace("<1", "0.5"),
                }
            )

        insights = sorted(
            insights,
            key=lambda item: abs(float(item.get("correlation", 0.0))) + abs(float(item.get("drift_correlation", 0.0))),
            reverse=True,
        )
        return {
            "lag_windows_seconds": list(self.config.lag_windows_seconds),
            "aggregation_frequency": self.config.aggregation_frequency,
            "signal_summaries": signal_summaries,
            "insights": insights[:5],
        }


def analyze_temporal_rca(events: pd.DataFrame, config: TemporalRCAConfig | None = None) -> dict[str, Any]:
    return TemporalRCAAnalyzer(config=config).analyze(events)