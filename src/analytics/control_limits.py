from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SIGNALS = ("temperature", "vibration", "speed")


@dataclass(frozen=True)
class ControlLimitConfig:
    warning_sigma: float = 2.0
    control_sigma: float = 3.0
    trend_window: int = 5


class ControlLimitAnalyzer:
    def __init__(self, config: ControlLimitConfig | None = None):
        self.config = config or ControlLimitConfig()

    def analyze(self, events: pd.DataFrame) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        for signal in SIGNALS:
            if signal not in events.columns:
                continue
            values = pd.to_numeric(events[signal], errors="coerce").dropna()
            if len(values) < 3:
                continue

            mean = float(values.mean())
            std = float(values.std(ddof=0) or 0.0)
            if std <= 0:
                std = max(abs(mean) * 0.01, 1e-6)

            safe_min = float(values.quantile(0.1))
            safe_max = float(values.quantile(0.9))
            warning_min = mean - (self.config.warning_sigma * std)
            warning_max = mean + (self.config.warning_sigma * std)
            control_min = mean - (self.config.control_sigma * std)
            control_max = mean + (self.config.control_sigma * std)

            outside_mask = (values < control_min) | (values > control_max)
            warning_mask = ((values < warning_min) | (values > warning_max)) & ~outside_mask
            rolling_mean = values.rolling(window=self.config.trend_window, min_periods=self.config.trend_window).mean()
            trend_violations = int(
                ((rolling_mean > warning_max) | (rolling_mean < warning_min)).fillna(False).sum()
            )

            metrics[signal] = {
                "safe_range": {"min": round(safe_min, 4), "max": round(safe_max, 4)},
                "warning_limits": {"min": round(warning_min, 4), "max": round(warning_max, 4)},
                "control_limits": {"min": round(control_min, 4), "max": round(control_max, 4)},
                "defect_rate_outside_control_limits": round(float(outside_mask.mean()), 4),
                "defect_rate_outside_warning_limits": round(float((outside_mask | warning_mask).mean()), 4),
                "trend_violations": trend_violations,
                "sample_size": int(len(values)),
            }

        return {"metrics": metrics}


def analyze_control_limits(events: pd.DataFrame, config: ControlLimitConfig | None = None) -> dict[str, Any]:
    return ControlLimitAnalyzer(config=config).analyze(events)