"""Per-cycle feature extraction from process curves.

Single scalar readings throw away most of the signal; defect precursors live in the
*shape* of the cycle (peak height, timing, plateau level, area). Every curve is
summarised into a fixed set of statistics, and - once a golden reference has been
learned from known-good cycles - into its deviation from the golden curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from src.process.models import ProcessCycle

CURVE_STATS = ("mean", "std", "max", "min", "time_to_peak_s", "integral", "final", "max_slope", "active_median")
GOLDEN_RESAMPLE_POINTS = 200


def _finite(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0


def curve_statistics(values: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {stat: 0.0 for stat in CURVE_STATS}
    dt = 1.0 / max(sample_rate_hz, 1e-6)
    peak_idx = int(np.argmax(arr))
    slope = np.diff(arr) / dt if arr.size > 1 else np.zeros(1)
    threshold = 0.1 * float(np.max(np.abs(arr))) if np.max(np.abs(arr)) > 0 else 0.0
    active = arr[np.abs(arr) > threshold]
    return {
        "mean": _finite(arr.mean()),
        "std": _finite(arr.std()),
        "max": _finite(arr.max()),
        "min": _finite(arr.min()),
        "time_to_peak_s": _finite(peak_idx * dt),
        "integral": _finite(np.trapezoid(arr, dx=dt)),
        "final": _finite(arr[-1]),
        "max_slope": _finite(np.max(slope)),
        # Level of the "working" part of the curve (e.g. the hold-pressure plateau),
        # independent of how long the idle phases before/after it last.
        "active_median": _finite(np.median(active) if active.size else 0.0),
    }


def resample(values: np.ndarray, points: int = GOLDEN_RESAMPLE_POINTS) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return np.zeros(points)
    if arr.size == 1:
        return np.full(points, arr[0])
    x_old = np.linspace(0.0, 1.0, arr.size)
    x_new = np.linspace(0.0, 1.0, points)
    return np.interp(x_new, x_old, arr)


@dataclass
class GoldenCycleReference:
    """Median curve shape of known-good cycles, per channel (time-normalised)."""

    curves: dict[str, np.ndarray] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    n_cycles: int = 0

    @classmethod
    def fit(cls, cycles: Iterable[ProcessCycle], min_cycles: int = 10) -> "GoldenCycleReference | None":
        cycles = list(cycles)
        if len(cycles) < min_cycles:
            return None
        stacked: dict[str, list[np.ndarray]] = {}
        for cycle in cycles:
            for name, values in cycle.curves.items():
                stacked.setdefault(name, []).append(resample(values))
        curves, scale = {}, {}
        for name, rows in stacked.items():
            matrix = np.vstack(rows)
            curves[name] = np.median(matrix, axis=0)
            # Normalise deviations by the typical good-cycle spread (robust, never 0)
            spread = float(np.median(np.abs(matrix - curves[name])))
            scale[name] = max(spread, 1e-6 + 1e-3 * float(np.max(np.abs(curves[name])) or 1.0))
        return cls(curves=curves, scale=scale, n_cycles=len(cycles))

    def deviation(self, name: str, values: np.ndarray) -> dict[str, float]:
        golden = self.curves.get(name)
        if golden is None:
            return {}
        diff = resample(values) - golden
        scale = self.scale.get(name, 1.0)
        return {
            "golden_rmse": _finite(np.sqrt(np.mean(diff ** 2)) / scale),
            "golden_max_dev": _finite(np.max(np.abs(diff)) / scale),
            "golden_bias": _finite(np.mean(diff) / scale),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "curves": {k: v.tolist() for k, v in self.curves.items()},
            "scale": dict(self.scale),
            "n_cycles": self.n_cycles,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GoldenCycleReference | None":
        if not payload or not payload.get("curves"):
            return None
        return cls(
            curves={k: np.asarray(v, dtype=np.float64) for k, v in payload["curves"].items()},
            scale={k: float(v) for k, v in (payload.get("scale") or {}).items()},
            n_cycles=int(payload.get("n_cycles", 0)),
        )


class CycleFeatureExtractor:
    """Turns a :class:`ProcessCycle` into a flat, numeric feature dictionary."""

    def __init__(self, golden: GoldenCycleReference | None = None):
        self.golden = golden

    def extract(self, cycle: ProcessCycle) -> dict[str, float]:
        features: dict[str, float] = {}
        for name, value in sorted(cycle.setpoints.items()):
            features[f"setpoint.{name}"] = _finite(value)
        for name, value in sorted(cycle.scalars.items()):
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                features[name] = _finite(value)
        features["cycle.duration_s"] = _finite(cycle.duration_s)
        for name, values in sorted(cycle.curves.items()):
            for stat, value in curve_statistics(values, cycle.sample_rate_hz).items():
                features[f"{name}.{stat}"] = value
            if self.golden is not None:
                for stat, value in self.golden.deviation(name, values).items():
                    features[f"{name}.{stat}"] = value
        return features
