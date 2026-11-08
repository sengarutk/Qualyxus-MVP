"""Core data model for process (machine) cycles and their quality outcomes.

A *cycle* is one production event that creates one part (or one shot of a
multi-cavity tool): an injection-molding shot, a weld, a press stroke. Each cycle
carries the setpoints that were active, scalar measurements, and the raw sensor
curves sampled during the cycle. Quality outcomes (vision verdicts, manual audits)
are linked to cycles after the fact through :mod:`src.process.alignment`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PASS = "pass"
FAIL = "fail"
NOT_INSPECTED = "not_inspected"   # nobody could judge this part: never a pass, never a training label


@dataclass
class ProcessCycle:
    cycle_id: str
    machine_id: str
    sequence: int
    start_ts: float
    end_ts: float
    setpoints: dict[str, float] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    sample_rate_hz: float = 50.0
    line_id: str | None = None
    source: str = "unknown"

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_ts) - float(self.start_ts))

    @staticmethod
    def make_id(machine_id: str, sequence: int) -> str:
        return f"{machine_id}-{int(sequence):09d}"


@dataclass
class InspectionRecord:
    inspection_id: str
    camera_id: str
    ts: float
    verdict: str
    defect_type: str | None = None
    anomaly_score: float | None = None
    event_id: str | None = None
    cycle_id: str | None = None
    alignment_confidence: float | None = None
    source: str = "vision"  # "vision" | "audit"
    pos_x: float | None = None        # part centre in the frame, pixels
    pos_y: float | None = None
    pos_angle: float | None = None    # degrees
    cavity: int = 0


@dataclass
class RiskAssessment:
    cycle_id: str
    risk: float
    decision: str  # "ok" | "watch" | "high"
    model_version: str
    model_kind: str  # "supervised" | "deviation"
    likely_defect: str | None = None
    likely_defect_probability: float | None = None
    top_factors: list[dict[str, Any]] = field(default_factory=list)
    ts: float = 0.0
