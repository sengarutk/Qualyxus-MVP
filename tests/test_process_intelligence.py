from __future__ import annotations

import numpy as np
import pytest

from src.process.models import FAIL, PASS
from src.process.prescriptions import PrescriptionEngine
from src.process.risk_model import DefectRiskModel, fit_deviation_model, is_better, train_supervised_model
from src.process.simulator import INJECTION_MOLDING_KNOWLEDGE, SETPOINT_LIMITS, InjectionMoldingSimulator


def _auc(model: DefectRiskModel, rows) -> float:
    from sklearn.metrics import roc_auc_score

    X = np.array([model.vectorize(r["features"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    return roc_auc_score(y, model.risk_matrix(X))


def test_supervised_model_generalises_out_of_time(simulated_rows):
    rows = simulated_rows(5000, seed=11)
    model = train_supervised_model(rows[:4000])
    assert model.kind == "supervised"
    assert model.metrics["validation"]["roc_auc"] > 0.75
    assert _auc(model, rows[4000:]) > 0.8          # later, unseen production period
    assert 0.0 < model.watch_threshold < model.high_threshold < 1.0


def test_supervised_model_refuses_insufficient_data(simulated_rows):
    with pytest.raises(ValueError):
        train_supervised_model(simulated_rows(100))


def test_deviation_model_flags_abnormal_cycles(simulated_rows):
    # Reference = a healthy start-up window; score the later production period with no labels.
    rows = simulated_rows(1500, seed=4)
    model = fit_deviation_model([r for r in rows[:200] if not r["label"]])
    assert _auc(model, rows[200:]) > 0.8
    good = next(r for r in rows if not r["label"])
    cold = dict(good["features"], **{"melt_temp.mean": good["features"]["melt_temp.mean"] - 25.0})
    assessment = model.assess("c1", cold)
    assert assessment.decision == "high"
    assert assessment.top_factors[0]["feature"].startswith("melt_temp")
    assert assessment.top_factors[0]["direction"] == "below"


def test_deviation_model_ignores_deliberate_setpoint_changes(simulated_rows):
    rows = simulated_rows(300, seed=4)
    model = fit_deviation_model([r for r in rows if not r["label"]])
    good = next(r for r in rows if not r["label"])
    changed = dict(good["features"], **{"setpoint.hold_pressure_sp": 800.0})
    assert model.assess("c", changed).decision == "ok"


def test_model_persistence_roundtrip(tmp_path, simulated_rows):
    rows = simulated_rows(1500, seed=11)
    model = train_supervised_model(rows, min_samples=500)
    loaded = DefectRiskModel.load(model.save(tmp_path))
    X = np.array([model.vectorize(r["features"]) for r in rows[:50]])
    np.testing.assert_allclose(model.risk_matrix(X), loaded.risk_matrix(X))


def test_promotion_rule():
    def fake(kind, auc):
        return DefectRiskModel(kind, "v", [], np.zeros(0), np.ones(0), 0.3, 0.6,
                               metrics={"validation": {"roc_auc": auc}} if auc is not None else {})
    assert is_better(fake("supervised", 0.8), fake("deviation", None))
    assert not is_better(fake("supervised", 0.6), None)              # barely better than chance
    # Compared on the same held-out window only:
    assert not is_better(fake("supervised", 0.75), fake("supervised", 0.9), incumbent_auc=0.85)
    assert is_better(fake("supervised", 0.84), fake("supervised", 0.95), incumbent_auc=0.85)


def test_prescription_raises_melt_temperature_for_cold_melt(tmp_path):
    from src.process.features import CycleFeatureExtractor

    sim = InjectionMoldingSimulator(seed=4, start_ts=1e6, disturbance_scale=0.0)
    extractor = CycleFeatureExtractor()
    healthy = []
    for _ in range(200):
        cycle, outcome = sim.next_cycle()
        healthy.append({"end_ts": cycle.end_ts, "features": extractor.extract(cycle), "label": 0})
    model = fit_deviation_model(healthy)
    sim.heater_offset = -12.0  # failed heater band: the real physical failure mode
    cycle, _ = sim.next_cycle()
    cold = extractor.extract(cycle)
    good = {"cycle": cycle}
    assessment = model.assess(cycle.cycle_id, cold)
    assert assessment.decision == "high"
    assert "melt_temp" in {f["group"] for f in assessment.top_factors}
    engine = PrescriptionEngine(INJECTION_MOLDING_KNOWLEDGE, SETPOINT_LIMITS, cooldown_cycles=10)
    setpoints = dict(good["cycle"].setpoints)
    out = engine.recommend(assessment, cold, setpoints, model, machine_id="imm_1", sequence=100)
    melt = [p for p in out if p.setpoint == "melt_temp_sp"]
    assert melt and melt[0].recommended > setpoints["melt_temp_sp"]
    lo, hi = SETPOINT_LIMITS["melt_temp_sp"]
    assert melt[0].delta <= 0.08 * (hi - lo) + 1e-9          # bounded step
    assert melt[0].risk_if_normal < assessment.risk
    # Cooldown: no repeat recommendation for the same setpoint right away.
    again = engine.recommend(assessment, cold, setpoints, model, machine_id="imm_1", sequence=105)
    assert not [p for p in again if p.setpoint == "melt_temp_sp"]

    from src.operations.action_tracker import ActionTracker

    tracker = ActionTracker(sqlite_path=str(tmp_path / "actions.db"))
    action = PrescriptionEngine.issue(tracker, melt[0])
    assert action["approval_required"] is True
    assert action["action_payload"]["action_type"] == "process_prescription"
    with pytest.raises(ValueError):  # must be acknowledged and approved before closing
        tracker.update_action(action["action_id"], status="closed", changed_by="x")


def test_service_flywheel_end_to_end(tmp_path):
    """Cycles -> provisional golden -> deviation model -> labels -> supervised promotion."""
    from src.process.service import ProcessIntelligenceService

    cfg = {
        "db_path": str(tmp_path / "process.db"),
        "models_dir": str(tmp_path / "models"),
        "model": {"golden_warmup_cycles": 30, "retrain_every_labels": 10**9,
                  "min_train_samples": 300, "min_train_positives": 20},
        "prescriptions": {"enabled": True},
    }
    cams = [{"id": "cam", "process_link": {"machine_id": "imm_1", "mode": "time",
                                            "transport_delay_s": 3.0, "tolerance_s": 2.0}}]
    service = ProcessIntelligenceService(cfg, cameras_cfg=cams)
    sim = InjectionMoldingSimulator(seed=11, start_ts=1_000_000.0)
    alerts = []
    service.add_alert_callback(lambda a, c, p: alerts.append((a, p)))

    truth = {}
    for i in range(1500):
        cycle, outcome = sim.next_cycle()
        service.on_cycle(cycle)
        truth[cycle.cycle_id] = outcome
        record = service.record_inspection(
            "cam", cycle.end_ts + 3.0 + 0.3, FAIL if outcome.is_defective else PASS,
            defect_type=outcome.defect_type,
        )
        assert record.cycle_id == cycle.cycle_id           # time alignment found the right cycle
        if i == 29:
            assert service.model is not None and service.model.kind == "deviation"

    report = service.retrain()
    assert report["promoted"] and service.model.kind == "supervised"
    assert service.store.active_model()["version"] == service.model.version

    # A fresh service reloads the active model and golden reference from disk.
    reloaded = ProcessIntelligenceService(cfg, cameras_cfg=cams)
    assert reloaded.model.version == service.model.version and reloaded.golden is not None
    assert any(a.decision != "ok" for a, _ in alerts)

    import time as _time

    kpis = service.kpis(window_s=_time.time())
    assert kpis["parts_inspected"] == 1500 and kpis["alignment_coverage"] == 1.0
    assert kpis["prediction"]["cycles_scored"] > 1000
    telemetry = service.telemetry_for(next(iter(truth)))
    assert telemetry["machine_id"] == "imm_1" and "temperature" in telemetry


def test_kpis_report_missing_ground_truth_as_none(tmp_path):
    from src.process.kpis import compute_kpis
    from src.process.store import ProcessStore

    k = compute_kpis(ProcessStore(tmp_path / "p.db"), 0.0, 1e12)
    assert k["scrap_rate"] is None and k["false_reject_rate"] is None and k["escape_rate"] is None


def test_spec_limits_add_risk_only_outside_the_window_and_drive_prescriptions():
    from src.process.features import CycleFeatureExtractor
    from src.process.simulator import INJECTION_MOLDING_SPEC_LIMITS

    sim = InjectionMoldingSimulator(seed=4, start_ts=1e6, disturbance_scale=0.0)
    extractor = CycleFeatureExtractor()
    healthy = []
    for _ in range(200):
        cycle, _ = sim.next_cycle()
        healthy.append({"end_ts": cycle.end_ts, "features": extractor.extract(cycle), "label": 0})
    model = fit_deviation_model(healthy)
    model.spec_limits = INJECTION_MOLDING_SPEC_LIMITS
    inside = np.array([model.vectorize(r["features"]) for r in healthy[-20:]])
    assert np.all(model.spec_risk(inside) == 0.0)

    sim.dryer_fault = 0.12  # wet material: dew point far above the datasheet limit
    cycle, _ = sim.next_cycle()
    wet = extractor.extract(cycle)
    assert wet["dryer_dewpoint_c"] > INJECTION_MOLDING_SPEC_LIMITS["dryer_dewpoint_c"]["max"]
    assessment = model.assess(cycle.cycle_id, wet)
    assert assessment.decision == "high"
    out = PrescriptionEngine(INJECTION_MOLDING_KNOWLEDGE, SETPOINT_LIMITS).recommend(
        assessment, wet, dict(cycle.setpoints), model, machine_id="imm_1", sequence=1)
    speed = [p for p in out if p.setpoint == "injection_speed_sp"]
    assert speed and speed[0].recommended < cycle.setpoints["injection_speed_sp"]


def test_service_rebaselines_after_intentional_setpoint_change(tmp_path):
    from src.process.service import ProcessIntelligenceService

    cfg = {"db_path": str(tmp_path / "p.db"), "models_dir": str(tmp_path / "m"),
           "model": {"golden_warmup_cycles": 30, "retrain_every_labels": 10**9, "regime_min_good_cycles": 20}}
    cams = [{"id": "cam", "process_link": {"machine_id": "imm_1", "transport_delay_s": 0.0, "tolerance_s": 1.0}}]
    service = ProcessIntelligenceService(cfg, cameras_cfg=cams)
    sim = InjectionMoldingSimulator(seed=3, start_ts=1e6, disturbance_scale=0.0)

    def run(n):
        decisions = []
        for _ in range(n):
            cycle, outcome = sim.next_cycle()
            a = service.on_cycle(cycle)
            service.record_inspection("cam", cycle.end_ts, FAIL if outcome.is_defective else PASS)
            decisions.append(a.decision if a else None)
        return decisions

    run(80)
    steady = service._current_reference(service.model)
    assert steady is not None and not steady.exclude_groups  # verified-good baseline, nothing excluded
    sim.apply_setpoints({"hold_pressure_sp": 700.0})  # an approved prescription
    after = run(5)
    ref = service._current_reference(service.model)
    assert ref is not None and "hold_pressure" in ref.exclude_groups and ref.median is None
    assert after.count("high") == 0          # the intended change is not an alarm
    run(25)
    ref = service._current_reference(service.model)
    assert ref is not None and ref.median is not None  # new regime baseline learned
