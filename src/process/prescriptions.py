"""Setpoint prescriptions derived from risk explanations.

The risk model says *which measured signals* are abnormal and how much risk each one
carries. A process-knowledge map (per process type, editable in config) says which
controllable setpoint moves each signal and in which direction. Together they produce a
bounded, explained recommendation such as::

    Raise hold_pressure_sp 600 -> 648 bar: hold_pressure.active_median is 528 (normal 590);
    predicted risk 0.81 -> 0.22 if it returns to normal.

Prescriptions are never written to the machine automatically: they are issued as
approval-gated actions so a process engineer stays in the loop until trust is earned.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from src.process.models import RiskAssessment
from src.process.risk_model import DefectRiskModel, DeviationReference

logger = logging.getLogger("QualyxusPrescriptions")


@dataclass
class Prescription:
    machine_id: str
    cycle_id: str
    setpoint: str
    current: float
    recommended: float
    driver_feature: str
    driver_value: float
    driver_normal: float
    risk_before: float
    risk_if_normal: float
    rationale: str
    created_ts: float = field(default_factory=time.time)

    @property
    def delta(self) -> float:
        return self.recommended - self.current

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["delta"] = round(self.delta, 4)
        return payload


class PrescriptionEngine:
    def __init__(
        self,
        knowledge: dict[str, dict[str, Any]],
        limits: dict[str, tuple[float, float]] | None = None,
        *,
        max_step_fraction: float = 0.08,
        min_risk_drop: float = 0.1,
        cooldown_cycles: int = 60,
        min_abs_z: float = 1.0,
    ):
        self.knowledge = dict(knowledge or {})
        self.limits = {k: (float(v[0]), float(v[1])) for k, v in (limits or {}).items()}
        self.max_step_fraction = float(max_step_fraction)
        self.min_risk_drop = float(min_risk_drop)
        self.cooldown_cycles = int(cooldown_cycles)
        self.min_abs_z = float(min_abs_z)
        self._last_issued: dict[tuple[str, str], int] = {}

    def _max_step(self, setpoint: str, current: float) -> float:
        if setpoint in self.limits:
            lo, hi = self.limits[setpoint]
            return self.max_step_fraction * (hi - lo)
        return self.max_step_fraction * max(abs(current), 1.0)

    def recommend(
        self,
        assessment: RiskAssessment,
        features: dict[str, float],
        setpoints: dict[str, float],
        model: DefectRiskModel,
        *,
        machine_id: str,
        sequence: int,
        reference: DeviationReference | None = None,
    ) -> list[Prescription]:
        """Rank every abnormal *controllable* driver by the risk its correction removes.

        Every driver in the knowledge map is evaluated (not only the top explanations: those
        may be dominated by downstream symptoms such as cavity pressure). The counterfactual
        returns the driver and its declared causal effects to normal together.
        """
        if assessment.decision != "high":
            return []
        candidates: list[tuple[float, str, dict[str, Any], float]] = []
        for feature, rule in self.knowledge.items():
            if feature not in features:
                continue
            z = model.feature_z(features, feature)  # vs. the healthy (model-wide) good state
            if z is None or abs(z) < self.min_abs_z:
                continue
            groups = list(rule.get("effects") or [DefectRiskModel.signal_group(feature)])
            overrides = model.normal_overrides_for_groups(groups)
            risk_if_normal = model.counterfactual_risk(features, overrides, reference)
            drop = float(assessment.risk) - risk_if_normal
            if drop >= self.min_risk_drop:
                candidates.append((drop, feature, rule, risk_if_normal))
        candidates.sort(key=lambda c: c[0], reverse=True)

        out: list[Prescription] = []
        used_setpoints: set[str] = set()
        for _drop, feature, rule, risk_if_normal in candidates:
            setpoint = str(rule["setpoint"])
            if setpoint in used_setpoints or setpoint not in setpoints:
                continue
            last = self._last_issued.get((machine_id, setpoint))
            if last is not None and sequence - last < self.cooldown_cycles:
                continue  # give the previous correction time to take effect

            value = float(features[feature])
            normal = model.normal_group_overrides(feature).get(feature, value)
            current = float(setpoints[setpoint])
            step = (normal - value) * float(rule.get("direction", 1.0)) * float(rule.get("gain", 1.0))
            limit = self._max_step(setpoint, current)
            step = max(-limit, min(limit, step))
            recommended = current + step
            if setpoint in self.limits:
                lo, hi = self.limits[setpoint]
                recommended = max(lo, min(hi, recommended))
            if abs(recommended - current) < 1e-6:
                continue

            verb = "Raise" if recommended > current else "Lower"
            rationale = (
                f"{verb} {setpoint} {current:g} -> {recommended:.4g}: {feature} is {value:.4g} "
                f"(normal {normal:.4g}); predicted risk {assessment.risk:.2f} -> {risk_if_normal:.2f} "
                f"once it returns to normal."
            )
            out.append(Prescription(
                machine_id=machine_id,
                cycle_id=assessment.cycle_id,
                setpoint=setpoint,
                current=current,
                recommended=round(recommended, 4),
                driver_feature=feature,
                driver_value=value,
                driver_normal=float(normal),
                risk_before=float(assessment.risk),
                risk_if_normal=round(float(risk_if_normal), 5),
                rationale=rationale,
            ))
            used_setpoints.add(setpoint)
            self._last_issued[(machine_id, setpoint)] = sequence
        return out

    @staticmethod
    def issue(tracker, prescription: Prescription, *, likely_defect: str | None = None) -> dict[str, Any]:
        """Create an approval-gated action in the ActionTracker for a prescription."""
        payload = {
            "title": f"Adjust {prescription.setpoint} on {prescription.machine_id}",
            "action_type": "process_prescription",
            "classification_6m": "Machine",
            "target_system": "machine",
            "explanation": prescription.rationale,
            "likely_defect": likely_defect,
            **prescription.to_payload(),
        }
        return tracker.issue_action(
            rca_reference=f"cycle:{prescription.cycle_id}",
            assigned_role="production",
            action_payload=payload,
            approval_required=True,
            target_value=float(prescription.recommended),
            changed_by="system:process_intelligence",
        )
