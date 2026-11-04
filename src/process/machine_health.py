"""Machine-health index per component, read from the same cycles that explain defects.

A component that wears moves a measured signal away from where it sat when the machine was
healthy. Each component here maps to one indicator signal. The index is how far that signal
has moved, smoothed over recent cycles and measured in baseline noise units (robust z), so
one noisy cycle cannot raise an alarm and a slow slide cannot hide. Setpoint changes are
removed where the setpoint is known, so an intentional change is not read as wear.

The default map is the injection-moulding template. Other processes use their own map via
``process.components``. It has been checked against the simulator's hidden wear, where the
links are known; on a real machine each mapping needs confirming against maintenance records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

INJECTION_MOLDING_COMPONENTS: dict[str, dict[str, Any]] = {
    "Heater bands": {"signal": "melt_temp.active_median", "setpoint": "setpoint.melt_temp_sp", "direction": -1.0},
    "Check ring": {"signal": "hold_pressure.active_median", "setpoint": "setpoint.hold_pressure_sp", "direction": -1.0},
    "Mould cooling": {"signal": "mold_temp.active_median", "setpoint": "setpoint.mold_temp_sp", "direction": 1.0},
    "Material dryer": {"signal": "dryer_dewpoint_c", "direction": 1.0},
}

WATCH_Z = 4.0
ACT_Z = 10.0
EWMA_ALPHA = 0.1
MIN_BASELINE = 20
TREND_WINDOW = 200


@dataclass
class ComponentHealth:
    component: str
    signal: str
    health: float                     # 100 = as healthy as the baseline, 0 = at the act level
    wear_index: float                 # smoothed robust z, positive = worse
    status: str                       # OK | WATCH | ACT
    cycles_to_act: int | None         # linear estimate from the recent trend
    history: list[float]


def _residual(rows: Sequence[Mapping[str, float]], spec: Mapping[str, Any]) -> np.ndarray:
    signal, setpoint = spec["signal"], spec.get("setpoint")
    values = []
    for row in rows:
        value = row.get(signal)
        if value is None:
            values.append(np.nan)
        else:
            values.append(float(value) - float(row.get(setpoint, 0.0) if setpoint else 0.0))
    return np.asarray(values, dtype=np.float64)


def _ewma(values: np.ndarray, alpha: float = EWMA_ALPHA) -> np.ndarray:
    out = np.empty_like(values)
    level = np.nan
    for i, value in enumerate(values):
        if np.isfinite(value):
            level = value if not np.isfinite(level) else alpha * value + (1.0 - alpha) * level
        out[i] = level
    return out


def assess_component(name: str, spec: Mapping[str, Any], baseline_rows: Sequence[Mapping[str, float]],
                     rows: Sequence[Mapping[str, float]]) -> ComponentHealth | None:
    """None when the machine does not report this signal or the baseline is too short."""
    baseline = _residual(baseline_rows, spec)
    baseline = baseline[np.isfinite(baseline)]
    recent = _residual(rows, spec)
    if baseline.size < MIN_BASELINE or not np.isfinite(recent).any():
        return None
    center = float(np.median(baseline))
    sigma = max(1.4826 * float(np.median(np.abs(baseline - center))), 1e-6)
    index = float(spec.get("direction", 1.0)) * (_ewma(recent) - center) / sigma
    current = float(index[np.isfinite(index)][-1])
    status = "ACT" if current >= ACT_Z else ("WATCH" if current >= WATCH_Z else "OK")

    cycles_to_act = None
    tail = index[np.isfinite(index)][-TREND_WINDOW:]
    if status != "ACT" and tail.size >= 30:
        slope = float(np.polyfit(np.arange(tail.size), tail, 1)[0])
        if slope > 1e-4:
            cycles_to_act = int((ACT_Z - current) / slope)
    return ComponentHealth(
        component=name, signal=str(spec["signal"]), wear_index=round(current, 2),
        health=round(100.0 * (1.0 - float(np.clip(current / ACT_Z, 0.0, 1.0))), 1),
        status=status, cycles_to_act=cycles_to_act,
        history=[round(float(v), 3) for v in index],
    )


def assess_machine(baseline_rows: Sequence[Mapping[str, float]], rows: Sequence[Mapping[str, float]],
                   components: Mapping[str, Mapping[str, Any]] | None = None) -> list[ComponentHealth]:
    specs = components or INJECTION_MOLDING_COMPONENTS
    results = (assess_component(name, spec, baseline_rows, rows) for name, spec in specs.items())
    return [r for r in results if r is not None]
