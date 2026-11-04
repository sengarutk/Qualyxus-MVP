"""Physics-inspired injection-molding process simulator.

This is a development and validation harness, not a digital twin. It produces
realistic per-cycle sensor curves whose *hidden* process disturbances (heater band
degradation, check-ring wear, mold cooling fouling, wet material) causally drive the
defects a camera would see. That lets the whole vision + telemetry pipeline
(alignment, labelling, risk prediction, prescriptions) be exercised and measured
end-to-end before real plant data exists.

Defect mechanisms modelled (each driven by a measurable process margin):

* ``short_shot`` - melt too cold -> viscosity up -> required fill pressure exceeds the
  injection pressure limit (visible as a flat-topped pressure curve).
* ``sink_mark``  - insufficient packing: worn check ring bleeds hold pressure, or a hot
  mold needs more packing.
* ``flash``      - melt too hot / too fluid -> cavity pressure exceeds the clamp limit.
* ``burn_mark``  - wet material + high injection speed -> trapped gas dieseling.

A small irreducible rate of random defects (contamination) is included so no model can
reach a perfect score on simulated data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.process.models import ProcessCycle

DEFECT_TYPES = ("short_shot", "sink_mark", "flash", "burn_mark")

DEFAULT_SETPOINTS: dict[str, float] = {
    "melt_temp_sp": 230.0,          # degC
    "mold_temp_sp": 60.0,           # degC
    "injection_pressure_sp": 1000.0,  # bar (machine pressure limit for filling)
    "injection_speed_sp": 80.0,     # mm/s
    "hold_pressure_sp": 600.0,      # bar
    "hold_time_sp": 4.0,            # s
    "cooling_time_sp": 12.0,        # s
}

SETPOINT_LIMITS: dict[str, tuple[float, float]] = {
    "melt_temp_sp": (200.0, 260.0),
    "mold_temp_sp": (40.0, 80.0),
    "injection_pressure_sp": (700.0, 1400.0),
    "injection_speed_sp": (40.0, 120.0),
    "hold_pressure_sp": (400.0, 900.0),
    "hold_time_sp": (2.0, 8.0),
    "cooling_time_sp": (8.0, 20.0),
}

# Default process-knowledge map used by the prescription engine: which setpoint moves
# which measured feature, and in which direction (+1 = raising the setpoint raises it).
INJECTION_MOLDING_KNOWLEDGE: dict[str, dict[str, Any]] = {
    # driver feature -> setpoint that corrects it.
    #   direction: +1 = raising the setpoint raises the feature
    #   gain:      setpoint units per unit of feature deviation from normal
    #   effects:   signal groups that return to normal once the driver is corrected
    #              (the causal downstream of the driver, e.g. cold melt -> high pressures)
    "melt_temp.active_median": {"setpoint": "melt_temp_sp", "direction": 1.0, "gain": 1.0,
                                "effects": ["melt_temp", "injection_pressure", "cavity_pressure"]},
    "hold_pressure.active_median": {"setpoint": "hold_pressure_sp", "direction": 1.0, "gain": 1.1,
                                    "effects": ["hold_pressure", "cushion_mm", "cavity_pressure", "injection_pressure"]},
    "cushion_mm": {"setpoint": "hold_pressure_sp", "direction": 1.0, "gain": 60.0,
                   "effects": ["hold_pressure", "cushion_mm", "cavity_pressure", "injection_pressure"]},
    "mold_temp.active_median": {"setpoint": "mold_temp_sp", "direction": 1.0, "gain": 1.0,
                                "effects": ["mold_temp"]},
    # Wet material (high dryer dew point) is compensated by slower injection.
    "dryer_dewpoint_c": {"setpoint": "injection_speed_sp", "direction": 1.0, "gain": 1.1,
                         "effects": ["dryer_dewpoint_c"]},
}


# Datasheet-style process windows for the simulated PA6-like material.
INJECTION_MOLDING_SPEC_LIMITS: dict[str, dict[str, float]] = {
    "dryer_dewpoint_c": {"max": -10.0, "softness": 5.0},      # wet material -> splay / burns
    "melt_temp.active_median": {"min": 215.0, "max": 250.0, "softness": 3.0},
    "mold_temp.active_median": {"min": 45.0, "max": 75.0, "softness": 3.0},
}


@dataclass
class SimulatedOutcome:
    """Ground truth known only to the simulator (used for rendering and scoring)."""

    cycle_id: str
    defect_type: str | None
    margins: dict[str, float]
    probabilities: dict[str, float]
    hidden_state: dict[str, float] = field(default_factory=dict)

    @property
    def is_defective(self) -> bool:
        return self.defect_type is not None


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class InjectionMoldingSimulator:
    """Deterministic (seeded) cycle-by-cycle injection molding process model."""

    def __init__(
        self,
        machine_id: str = "imm_1",
        *,
        seed: int = 7,
        start_ts: float | None = None,
        setpoints: dict[str, float] | None = None,
        sample_rate_hz: float = 50.0,
        disturbance_scale: float = 1.0,
        random_defect_rate: float = 0.003,
        line_id: str | None = "line_A",
    ):
        import time

        self.machine_id = machine_id
        self.line_id = line_id
        self.rng = np.random.default_rng(seed)
        self.sample_rate_hz = float(sample_rate_hz)
        self.disturbance_scale = float(disturbance_scale)
        self.random_defect_rate = float(random_defect_rate)
        self.setpoints = dict(DEFAULT_SETPOINTS)
        if setpoints:
            self.apply_setpoints(setpoints)
        self.sequence = 0
        self.clock = float(start_ts if start_ts is not None else time.time())

        # Hidden disturbance state
        self.heater_offset = 0.0         # degC (<= 0 while a heater band degrades)
        self.heater_rate = 0.0           # degC lost per cycle during an episode
        self.check_ring_wear = float(self.rng.uniform(0.0, 0.05))  # fraction of hold pressure lost
        self.cooling_fouling = 0.0       # degC added to the mold temperature
        self.lot_index = 0
        self.lot_moisture = 0.05         # % water content of the current material lot
        self.dryer_fault = 0.0           # extra moisture during dryer fault episodes
        self._new_lot()

    # ------------------------------------------------------------------ control

    def apply_setpoints(self, changes: dict[str, float]) -> dict[str, float]:
        """Apply (clipped) setpoint changes; returns the values actually applied."""
        applied = {}
        for name, value in (changes or {}).items():
            if name not in SETPOINT_LIMITS:
                continue
            lo, hi = SETPOINT_LIMITS[name]
            applied[name] = float(min(hi, max(lo, float(value))))
            self.setpoints[name] = applied[name]
        return applied

    # --------------------------------------------------------------- dynamics

    def _new_lot(self) -> None:
        self.lot_index += 1
        self.lot_moisture = float(self.rng.uniform(0.02, 0.10))
        self.material_lot = f"LOT-{1000 + self.lot_index}"

    def _evolve_disturbances(self) -> None:
        s = self.disturbance_scale
        rng = self.rng
        # Heater band degradation episodes, repaired by maintenance when badly off.
        if self.heater_rate == 0.0 and rng.random() < 0.0015 * s:
            self.heater_rate = float(rng.uniform(0.03, 0.09))
        if self.heater_rate > 0.0:
            self.heater_offset -= self.heater_rate
            if self.heater_offset < -14.0 or rng.random() < 0.003:
                self.heater_offset, self.heater_rate = 0.0, 0.0  # maintenance replaced the band
        # Check ring wear accumulates slowly; replaced when very worn.
        self.check_ring_wear += float(abs(rng.normal(0.0002, 0.0001))) * s
        if self.check_ring_wear > 0.24:
            self.check_ring_wear = float(rng.uniform(0.0, 0.03))
        # Mold cooling channel fouling; cleaned periodically.
        self.cooling_fouling += float(abs(rng.normal(0.008, 0.004))) * s
        if self.cooling_fouling > 10.0 or rng.random() < 0.0008:
            self.cooling_fouling = 0.0
        # Material lots and dryer faults.
        if self.sequence % 300 == 0:
            self._new_lot()
        if self.dryer_fault == 0.0 and rng.random() < 0.0008 * s:
            self.dryer_fault = float(rng.uniform(0.06, 0.14))
        elif self.dryer_fault > 0.0 and rng.random() < 0.02:
            self.dryer_fault = 0.0

    # ------------------------------------------------------------------ cycle

    def next_cycle(self, start_ts: float | None = None) -> tuple[ProcessCycle, SimulatedOutcome]:
        rng = self.rng
        self.sequence += 1
        self._evolve_disturbances()
        sp = self.setpoints

        melt_temp = sp["melt_temp_sp"] + self.heater_offset + rng.normal(0.0, 0.8)
        mold_temp = sp["mold_temp_sp"] + self.cooling_fouling * (12.0 / sp["cooling_time_sp"]) + rng.normal(0.0, 0.6)
        speed = sp["injection_speed_sp"] * (1.0 + rng.normal(0.0, 0.01))
        moisture = max(0.0, self.lot_moisture + self.dryer_fault + rng.normal(0.0, 0.004))
        dewpoint = -40.0 + 300.0 * (moisture - 0.02) + rng.normal(0.0, 1.0)

        viscosity = math.exp(0.025 * (230.0 - melt_temp))
        required_fill = 780.0 * viscosity * (speed / 80.0) ** 0.3 * (1.0 + rng.normal(0.0, 0.012))
        peak_injection = min(required_fill, sp["injection_pressure_sp"])
        fill_ratio = min(1.0, sp["injection_pressure_sp"] / required_fill)

        hold_actual = sp["hold_pressure_sp"] * (1.0 - self.check_ring_wear) + rng.normal(0.0, 5.0)
        cushion = 5.0 - 9.0 * self.check_ring_wear + rng.normal(0.0, 0.12)

        pack_index = (hold_actual * min(sp["hold_time_sp"], 6.0)) / (600.0 * 4.0)
        pack_need = 0.72 * (1.0 + 0.012 * (mold_temp - 60.0)) * (1.0 + 0.01 * (12.0 - sp["cooling_time_sp"]))
        peak_cavity = (0.55 * peak_injection + 0.30 * hold_actual) * fill_ratio
        flash_index = peak_cavity / viscosity
        burn_index = (speed / 80.0) * (1.0 + 6.0 * moisture)

        margins = {
            "short_shot": sp["injection_pressure_sp"] / required_fill - 1.0,
            "sink_mark": pack_index / pack_need - 1.0,
            "flash": 1.0 - flash_index / 760.0,
            "burn_mark": 1.0 - burn_index / 1.9,
        }
        probabilities = {name: _sigmoid(-m / 0.03) for name, m in margins.items()}

        realised = [name for name, p in probabilities.items() if rng.random() < p]
        defect = max(realised, key=lambda n: probabilities[n]) if realised else None
        if defect is None and rng.random() < self.random_defect_rate:
            defect = str(rng.choice(DEFECT_TYPES))  # contamination: not predictable from process data

        fill_time = 0.9 * (80.0 / speed)
        hold_time = sp["hold_time_sp"]
        cool_time = sp["cooling_time_sp"]
        curves = self._curves(
            fill_time, hold_time, cool_time, peak_injection, sp["injection_pressure_sp"],
            fill_ratio, hold_actual, peak_cavity, melt_temp, mold_temp,
        )

        start = float(start_ts if start_ts is not None else self.clock)
        duration = fill_time + hold_time + cool_time
        end = start + duration
        self.clock = end + 1.5  # mold open / eject / close

        cycle = ProcessCycle(
            cycle_id=ProcessCycle.make_id(self.machine_id, self.sequence),
            machine_id=self.machine_id,
            line_id=self.line_id,
            sequence=self.sequence,
            start_ts=start,
            end_ts=end,
            setpoints=dict(sp),
            scalars={
                "cushion_mm": round(float(cushion), 4),
                "dryer_dewpoint_c": round(float(dewpoint), 3),
                "cycle_time_s": round(float(duration), 4),
                "material_lot": self.material_lot,
            },
            curves=curves,
            sample_rate_hz=self.sample_rate_hz,
            source="simulator",
        )
        outcome = SimulatedOutcome(
            cycle_id=cycle.cycle_id,
            defect_type=defect,
            margins={k: round(float(v), 5) for k, v in margins.items()},
            probabilities={k: round(float(v), 5) for k, v in probabilities.items()},
            hidden_state={
                "heater_offset": round(self.heater_offset, 4),
                "check_ring_wear": round(self.check_ring_wear, 5),
                "cooling_fouling": round(self.cooling_fouling, 4),
                "moisture": round(moisture, 5),
            },
        )
        return cycle, outcome

    def _curves(
        self, fill_t, hold_t, cool_t, peak_inj, inj_limit, fill_ratio, hold_p, peak_cav, melt, mold,
    ) -> dict[str, np.ndarray]:
        rng = self.rng
        fs = self.sample_rate_hz
        n_fill = max(2, int(round(fill_t * fs)))
        n_hold = max(2, int(round(hold_t * fs)))
        n_cool = max(2, int(round(cool_t * fs)))
        t_fill = np.linspace(0.0, 1.0, n_fill)
        t_hold = np.linspace(0.0, 1.0, n_hold)
        t_cool = np.linspace(0.0, 1.0, n_cool)

        # Hydraulic injection pressure: ramps while filling (flat-topped when the
        # machine pressure limit is reached), holds, then relaxes during cooling.
        fill_curve = peak_inj * t_fill ** 1.5
        if fill_ratio < 1.0:
            fill_curve = np.minimum(fill_curve * 1.15, inj_limit)
        inj = np.concatenate([
            fill_curve,
            hold_p * (1.0 - 0.04 * t_hold),
            hold_p * 0.96 * np.exp(-6.0 * t_cool),
        ])
        # Cavity pressure: lags the hydraulic pressure, decays as the part freezes.
        cav = np.concatenate([
            peak_cav * np.clip((t_fill - 0.3) / 0.7, 0.0, 1.0) ** 2,
            peak_cav * (0.55 + 0.45 * np.exp(-3.0 * t_hold)),
            peak_cav * 0.55 * np.exp(-4.0 * t_cool),
        ])
        n_total = n_fill + n_hold + n_cool
        melt_curve = melt + rng.normal(0.0, 0.35, n_total)
        bump = np.concatenate([np.zeros(n_fill), 3.0 * np.exp(-4.0 * t_hold), 0.6 * np.exp(-2.0 * t_cool)])
        mold_curve = mold + bump + rng.normal(0.0, 0.2, n_total)
        inj = inj + rng.normal(0.0, 3.0, n_total)
        cav = np.maximum(0.0, cav + rng.normal(0.0, 2.0, n_total))
        hold_curve = np.concatenate([np.zeros(n_fill), hold_p + rng.normal(0.0, 2.0, n_hold), np.zeros(n_cool)])
        return {
            "injection_pressure": inj.astype(np.float32),
            "cavity_pressure": cav.astype(np.float32),
            "hold_pressure": hold_curve.astype(np.float32),
            "melt_temp": melt_curve.astype(np.float32),
            "mold_temp": mold_curve.astype(np.float32),
        }
