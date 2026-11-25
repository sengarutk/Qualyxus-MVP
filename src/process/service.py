"""Process-quality intelligence service: the vision <-> telemetry flywheel.

Lifecycle of a part:

1. A machine cycle completes -> :meth:`on_cycle` extracts features against the golden
   cycle, stores the cycle, scores its defect risk *before the part reaches the camera*,
   and raises alerts / approval-gated prescriptions for high-risk cycles.
2. The camera inspects the part -> :meth:`record_inspection` links the verdict to the
   producing cycle (:mod:`src.process.alignment`). Every inspection - pass or fail - is a
   training label for the process model.
3. Periodically the risk model is retrained on the labelled cycles and promoted only if
   it beats the incumbent on time-ordered held-out data.
"""
from __future__ import annotations

import collections
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from src.process.alignment import AlignmentResult, CycleAligner, StationLink
from src.process.features import CycleFeatureExtractor, GoldenCycleReference
from src.process.kpis import compute_kpis
from src.process.models import FAIL, InspectionRecord, NOT_INSPECTED, PASS, ProcessCycle, RiskAssessment
from src.config.schema import required_views
from src.process.machine_health import INJECTION_MOLDING_COMPONENTS, assess_machine
from src.process.prescriptions import Prescription, PrescriptionEngine, withdraw
from src.process.risk_model import (
    DefectRiskModel,
    DeviationReference,
    evaluate_auc,
    fit_deviation_model,
    is_better,
    train_supervised_model,
    validation_rows,
)
from src.process.simulator import INJECTION_MOLDING_KNOWLEDGE, INJECTION_MOLDING_SPEC_LIMITS, SETPOINT_LIMITS
from src.process.store import PROCESS_CHANGE, ProcessStore

logger = logging.getLogger("QualyxusProcessIntelligence")

PRUNE_EVERY_CYCLES = 500
HEALTH_EVERY_CYCLES = 20

AlertCallback = Callable[[RiskAssessment, ProcessCycle, list[Prescription]], None]

# Telemetry fields carried on DefectEvent records, resolved from the producing cycle.
_TELEMETRY_FEATURES = {
    "temperature": ("melt_temp.mean", "melt_temp"),
    "speed": ("setpoint.injection_speed_sp", "injection_speed"),
}


@dataclass
class MachineState:
    """What the service has learned about one machine. Machines never share a baseline."""

    machine_id: str
    golden: GoldenCycleReference | None = None
    extractor: CycleFeatureExtractor = field(default_factory=CycleFeatureExtractor)
    model: DefectRiskModel | None = None
    provisional_good: list[ProcessCycle] = field(default_factory=list)
    reset_ts: float = 0.0            # newest process change already applied
    last_sequence: int | None = None
    watched: dict[str, Any] = field(default_factory=dict)     # values of baseline_reset_on scalars
    worn: dict[str, Any] = field(default_factory=dict)        # component -> ComponentHealth
    cycles: int = 0
    health_at: int = -10**9
    labels_since_train: int = 0

    # Operating regime = the active setpoint combination. Intentional setpoint changes
    # move measured signals by design; deviation scoring is re-baselined per regime.
    regime_setpoints: dict[str, float] | None = None
    regime_changed_groups: frozenset[str] = frozenset()
    regime_pending: "collections.OrderedDict[str, dict[str, float]]" = field(default_factory=collections.OrderedDict)
    regime_good: list[dict[str, float]] = field(default_factory=list)
    regime_reference: DeviationReference | None = None
    regime_reference_size: int = -1

    def clear_regime(self) -> None:
        self.regime_pending.clear()
        self.regime_good = []
        self.regime_reference, self.regime_reference_size = None, -1


class ProcessIntelligenceService:
    def __init__(
        self,
        process_cfg: dict[str, Any] | None = None,
        *,
        cameras_cfg: list[dict[str, Any]] | None = None,
        store: ProcessStore | None = None,
        action_tracker=None,
    ):
        cfg = dict(process_cfg or {})
        self.cfg = cfg
        self.store = store or ProcessStore(
            cfg.get("db_path", "data/events/process.db"), store_curves=bool(cfg.get("store_curves", True)),
        )
        self.curve_retention_s = float(cfg.get("curve_retention_days", 30)) * 86400.0
        self._cycles_seen = 0
        self.models_dir = Path(cfg.get("models_dir", "models/process"))
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.required = tuple(required_views(cameras_cfg or []))
        self.default_machine = str((cfg.get("machines") or [{}])[0].get("machine_id") or cfg.get("machine_id", "imm_1"))
        self.components = cfg.get("components") or INJECTION_MOLDING_COMPONENTS
        self.baseline_reset_on = list(cfg.get("baseline_reset_on") or [])

        links = []
        for cam in cameras_cfg or []:
            link_cfg = cam.get("process_link")
            if link_cfg:
                links.append(StationLink.from_config(str(cam.get("id")), link_cfg))
        self.aligner = CycleAligner(self.store, links)

        model_cfg = cfg.get("model", {}) or {}
        self.golden_warmup_cycles = int(model_cfg.get("golden_warmup_cycles", 50))
        self.retrain_every = int(model_cfg.get("retrain_every_labels", 250))
        self.min_train_samples = int(model_cfg.get("min_train_samples", 300))
        self.min_train_positives = int(model_cfg.get("min_train_positives", 20))
        self.max_train_cycles = int(model_cfg.get("max_train_cycles", 20000))
        self.min_promotion_auc = float(model_cfg.get("min_promotion_auc", 0.65))
        self.min_label_confidence = float(model_cfg.get("min_label_confidence", 0.5))
        self.min_regime_good = int(model_cfg.get("regime_min_good_cycles", 30))
        self.hybrid_deviation = bool(model_cfg.get("hybrid_deviation", False))
        spec = cfg.get("spec_limits")
        self.spec_limits = dict(INJECTION_MOLDING_SPEC_LIMITS if spec is None else spec)

        presc_cfg = cfg.get("prescriptions", {}) or {}
        self.prescriptions_enabled = bool(presc_cfg.get("enabled", True))
        limits = {k: tuple(v) for k, v in (presc_cfg.get("limits") or SETPOINT_LIMITS).items()}
        self.knowledge = presc_cfg.get("knowledge") or INJECTION_MOLDING_KNOWLEDGE
        self.prescriber = PrescriptionEngine(
            self.knowledge,
            limits,
            max_step_fraction=float(presc_cfg.get("max_step_fraction", 0.08)),
            min_risk_drop=float(presc_cfg.get("min_risk_drop", 0.1)),
            cooldown_cycles=int(presc_cfg.get("cooldown_cycles", 60)),
        )
        self._tracker = action_tracker

        self._lock = threading.RLock()
        self._retrain_lock = threading.Lock()
        self._alert_callbacks: list[AlertCallback] = []
        self._counted: "collections.OrderedDict[str, None]" = collections.OrderedDict()
        self._views_ok: "collections.OrderedDict[str, set[tuple[str, int]]]" = collections.OrderedDict()
        self.sources: list[Any] = []
        self._machines: dict[str, MachineState] = {}
        self._state(self.default_machine)

    # -------------------------------------------------------------- machines

    def _state(self, machine_id: str) -> MachineState:
        with self._lock:
            state = self._machines.get(machine_id)
            if state is None:
                state = MachineState(machine_id)
                state.last_sequence = self.store.last_sequence(machine_id)
                for last in self.store.recent_cycles(1, machine_id=machine_id):   # a change made while we were off
                    values = {**last.setpoints, **last.scalars}
                    state.watched = {n: values[n] for n in self.baseline_reset_on if values.get(n) is not None}
                state.reset_ts = self.store.baseline_start(machine_id, PROCESS_CHANGE)
                state.golden = self._load_golden(machine_id, state.reset_ts)
                state.extractor = CycleFeatureExtractor(state.golden)
                state.model = self._load_active_model(machine_id, state.reset_ts)
                if state.model is not None:
                    state.model.spec_limits = self.spec_limits
                    state.model.hybrid_deviation = self.hybrid_deviation
                self._machines[machine_id] = state
            return state

    # The first configured machine, for a station that has only one.
    @property
    def model(self) -> DefectRiskModel | None:
        return self._state(self.default_machine).model

    @property
    def golden(self) -> GoldenCycleReference | None:
        return self._state(self.default_machine).golden

    @property
    def extractor(self) -> CycleFeatureExtractor:
        return self._state(self.default_machine).extractor

    @property
    def source(self):
        return self.sources[0] if self.sources else None

    @property
    def _labels_since_train(self) -> int:
        return self._state(self.default_machine).labels_since_train

    # ----------------------------------------------------------- persistence

    def _golden_path(self, machine_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in machine_id)
        return self.models_dir / f"golden_cycle.{safe}.json"

    def _load_golden(self, machine_id: str, since_ts: float) -> GoldenCycleReference | None:
        paths = [self._golden_path(machine_id)]
        if machine_id == self.default_machine:
            paths.append(self.models_dir / "golden_cycle.json")    # saved before references were per machine
        for path in paths:
            try:
                if path.exists() and path.stat().st_mtime >= since_ts:
                    return GoldenCycleReference.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except Exception as e:
                logger.warning("[Process] Ignoring unreadable golden reference %s: %s", path, e)
        return None

    def _save_golden(self, machine_id: str, golden: GoldenCycleReference) -> None:
        path = self._golden_path(machine_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(golden.to_dict()), encoding="utf-8")
        tmp.replace(path)

    def _load_active_model(self, machine_id: str, since_ts: float) -> DefectRiskModel | None:
        entry = self.store.active_model(machine_id, any_machine=machine_id == self.default_machine)
        if not entry or entry["created_ts"] < since_ts:
            return None
        try:
            model = DefectRiskModel.load(entry["path"])
            logger.info("[Process] Loaded active %s risk model %s", model.kind, model.version)
            return model
        except Exception as e:
            logger.warning("[Process] Active model %s could not be loaded: %s", entry.get("version"), e)
            return None

    @property
    def _actions_path(self) -> Path:
        return Path(self.store.db_path).parent / "actions.db"

    @property
    def tracker(self):
        if self._tracker is None:
            from src.operations.action_tracker import ActionTracker

            self._tracker = ActionTracker(
                sqlite_path=str(self._actions_path)
            )
        return self._tracker

    # -------------------------------------------------------------- wiring

    def add_alert_callback(self, callback: AlertCallback) -> None:
        self._alert_callbacks.append(callback)

    def attach_source(self, source) -> None:
        self.sources.append(source)
        source.subscribe(self.on_cycle)

    def start(self) -> None:
        for source in self.sources:
            source.start()

    def stop(self) -> None:
        for source in self.sources:
            source.stop()

    # ------------------------------------------------------------ ingestion

    def on_cycle(self, cycle: ProcessCycle) -> RiskAssessment | None:
        with self._lock:
            state = self._state(cycle.machine_id)
            self._record_lost_cycles(state, cycle)
            self._apply_resets(state, cycle)
            if state.golden is None and cycle.complete:
                self._bootstrap_golden(state, cycle)
            features = state.extractor.extract(cycle)
            self.store.save_cycle(cycle, features)
            self._cycles_seen += 1
            state.cycles += 1
            if self.curve_retention_s > 0 and self._cycles_seen % PRUNE_EVERY_CYCLES == 0:
                self.store.prune_curves(self.curve_retention_s)
            if not cycle.complete:
                return None      # part of it was never sampled: it links its part and nothing more
            model = state.model
            self._track_regime(state, cycle, features)
            reference = self._current_reference(model, state)

        if model is None:
            return None
        assessment = model.assess(cycle.cycle_id, features, reference=reference)
        self.store.save_prediction(assessment)
        prescriptions: list[Prescription] = []
        if assessment.decision == "high" and self.prescriptions_enabled:
            # Corrections assume a sound machine. On a worn one they hide the wear, or pull against it
            # (in simulation they raised scrap), so the repair comes first.
            worn = self._worn_components(state, cycle)
            prescriptions = [] if worn else self.prescriber.recommend(
                assessment, features, cycle.setpoints, model,
                machine_id=cycle.machine_id, sequence=cycle.sequence, reference=reference,
            )
            for presc in prescriptions:
                try:
                    withdraw(self.tracker, presc.machine_id, "replaced by a newer suggestion",
                             setpoints=[presc.setpoint])
                    PrescriptionEngine.issue(self.tracker, presc, likely_defect=assessment.likely_defect)
                except Exception as e:
                    logger.error("[Process] Failed to issue prescription for %s: %s", cycle.cycle_id, e)
        if assessment.decision != "ok":
            for callback in list(self._alert_callbacks):
                try:
                    callback(assessment, cycle, prescriptions)
                except Exception as e:
                    logger.error("[Process] Alert callback failed: %s", e)
        return assessment

    def _record_lost_cycles(self, state: MachineState, cycle: ProcessCycle) -> None:
        last, state.last_sequence = state.last_sequence, int(cycle.sequence)
        if last is None or cycle.sequence <= last + 1:     # backwards is a counter reset: nothing is known
            return
        logger.error("[Process] %s: %d machine cycles have no data (%d to %d).", cycle.machine_id,
                     cycle.sequence - last - 1, last + 1, cycle.sequence - 1)
        self.store.save_gap(cycle.machine_id, cycle.end_ts, last + 1, cycle.sequence - 1)

    def _apply_resets(self, state: MachineState, cycle: ProcessCycle) -> None:
        for name in self.baseline_reset_on:
            value = cycle.scalars.get(name, cycle.setpoints.get(name))
            if value is None:
                continue
            before = state.watched.get(name, value)
            state.watched[name] = value
            if value != before:
                self.store.reset_baseline(cycle.machine_id, kind=PROCESS_CHANGE, by="system", ts=cycle.end_ts,
                                          reason=f"{name} changed from {before} to {value}")
        reset = self.store.baseline_start(cycle.machine_id, PROCESS_CHANGE)
        if reset > state.reset_ts:
            logger.warning("[Process] %s: the process changed, so the baseline and risk model start again.",
                           cycle.machine_id)
            state.reset_ts, state.golden, state.model = reset, None, None
            state.extractor = CycleFeatureExtractor()
            state.provisional_good, state.worn, state.labels_since_train = [], {}, 0
            state.regime_setpoints, state.regime_changed_groups = None, frozenset()
            state.clear_regime()
            self._withdraw(state.machine_id, "the mould, material or process changed")

    def _withdraw(self, machine_id: str, reason: str, **which) -> None:
        if not (self.prescriptions_enabled and (self._tracker is not None or self._actions_path.exists())):
            return
        try:
            withdraw(self.tracker, machine_id, reason, **which)
        except Exception as e:
            logger.error("[Process] Open suggestions could not be withdrawn: %s", e)

    def _worn_components(self, state: MachineState, cycle: ProcessCycle) -> set[str]:
        """Components drifting from their baseline. Each gets one maintenance request."""
        if state.cycles - state.health_at >= HEALTH_EVERY_CYCLES:
            state.health_at = state.cycles
            baseline, recent = self.store.health_rows(cycle.machine_id, recent=200)
            health = assess_machine(baseline, recent, self.components)
            state.worn = {c.component: c for c in health if c.status != "OK"}
        try:
            for name, c in state.worn.items():
                PrescriptionEngine.issue_maintenance(self.tracker, machine_id=cycle.machine_id, component=name,
                                                     cycle_id=cycle.cycle_id, status=c.status, wear=c.wear_index)
            if state.worn:
                withdraw(self.tracker, cycle.machine_id, f"{', '.join(state.worn)} is drifting: check it first")
        except Exception as e:
            logger.error("[Process] Maintenance request on %s failed: %s", cycle.machine_id, e)
        return set(state.worn)

    def _track_regime(self, state: MachineState, cycle: ProcessCycle, features: dict[str, float]) -> None:
        setpoints = {k: float(v) for k, v in (cycle.setpoints or {}).items()}
        if state.regime_setpoints is None:
            state.regime_setpoints = setpoints
        elif setpoints != state.regime_setpoints:
            changed = {k for k in set(setpoints) | set(state.regime_setpoints)
                       if abs(setpoints.get(k, float("nan")) - state.regime_setpoints.get(k, float("nan"))) > 1e-9
                       or (k in setpoints) != (k in state.regime_setpoints)}
            groups = set(state.regime_changed_groups)
            for driver, rule in self.knowledge.items():
                if rule.get("setpoint") in changed:
                    groups.update(rule.get("effects") or [DefectRiskModel.signal_group(driver)])
            logger.info("[Process] Operating regime changed (%s); re-baselining %s",
                        ", ".join(sorted(changed)), ", ".join(sorted(groups)) or "nothing")
            state.regime_setpoints = setpoints
            state.regime_changed_groups = frozenset(groups)
            state.clear_regime()
            # A suggestion made for settings that no longer apply is not advice any more.
            self._withdraw(state.machine_id, "the setting was changed", setpoints=changed)
        state.regime_pending[cycle.cycle_id] = features
        while len(state.regime_pending) > 500:
            state.regime_pending.popitem(last=False)

    def _current_reference(self, model: DefectRiskModel | None, state: MachineState | None = None) -> DeviationReference | None:
        if model is None:
            return None
        state = state or self._state(self.default_machine)
        if len(state.regime_good) >= self.min_regime_good:
            # Refresh the regime baseline as verified-good cycles accumulate.
            if state.regime_reference is None or len(state.regime_good) - state.regime_reference_size >= 10:
                state.regime_reference = model.reference_from_good(state.regime_good[-500:])
                state.regime_reference_size = len(state.regime_good)
            return state.regime_reference
        if state.regime_changed_groups:
            return DeviationReference(exclude_groups=state.regime_changed_groups)
        return None

    def _bootstrap_golden(self, state: MachineState, cycle: ProcessCycle) -> None:
        """Before any inspection exists, assume the first cycles of a run are healthy.

        The provisional reference is replaced by one built from *verified* good cycles on
        the first retrain, so a process that starts unhealthy self-corrects.
        """
        state.provisional_good.append(cycle)
        if len(state.provisional_good) < self.golden_warmup_cycles:
            return
        golden = GoldenCycleReference.fit(state.provisional_good, min_cycles=self.golden_warmup_cycles)
        if golden is None:
            return
        state.golden = golden
        state.extractor = CycleFeatureExtractor(golden)
        self._save_golden(state.machine_id, golden)
        rows = [{"end_ts": c.end_ts, "features": state.extractor.extract(c)} for c in state.provisional_good]
        try:
            self._activate(state, fit_deviation_model(rows, min_rows=min(20, len(rows))), n_samples=len(rows), n_positive=0)
        except ValueError as e:
            logger.warning("[Process] Deviation bootstrap skipped: %s", e)
        state.provisional_good = []
        logger.info("[Process] Provisional golden cycle learned from %d cycles", golden.n_cycles)

    # ---------------------------------------------------------- inspections

    def record_inspection(
        self,
        camera_id: str,
        ts: float,
        verdict: str,
        *,
        defect_type: str | None = None,
        anomaly_score: float | None = None,
        event_id: str | None = None,
        part_id: str | None = None,
        source: str = "vision",
        alignment: AlignmentResult | None = None,
        position: tuple[float, float, float] | None = None,
    ) -> InspectionRecord:
        if verdict not in (PASS, FAIL, NOT_INSPECTED):
            raise ValueError(f"verdict must be '{PASS}', '{FAIL}' or '{NOT_INSPECTED}', got {verdict!r}")
        if alignment is None:
            alignment = self.aligner.align(camera_id, ts, part_id=part_id)
        record = self.store.save_inspection(InspectionRecord(
            inspection_id=uuid.uuid4().hex,
            camera_id=camera_id,
            ts=float(ts),
            verdict=verdict,
            defect_type=defect_type if verdict == FAIL else None,
            anomaly_score=anomaly_score,
            event_id=event_id,
            cycle_id=alignment.cycle_id,
            alignment_confidence=alignment.confidence if alignment.cycle_id else None,
            source=source,
            pos_x=position[0] if position else None,
            pos_y=position[1] if position else None,
            pos_angle=position[2] if position else None,
            cavity=alignment.cavity,
        ))
        if alignment.cycle_id and verdict != NOT_INSPECTED:   # an unjudged part teaches nothing
            with self._lock:
                state = self._state_of_cycle(alignment.cycle_id)
                if alignment.cycle_id not in self._counted:     # one label per cycle, however many views
                    self._counted[alignment.cycle_id] = None
                    while len(self._counted) > 4096:
                        self._counted.popitem(last=False)
                    state.labels_since_train += 1
                self._track_good(state, alignment, camera_id, verdict, source)
                due = state.labels_since_train >= self.retrain_every
            if due:
                self.retrain_async(state.machine_id)
        elif alignment.cycle_id:
            with self._lock:
                self._state_of_cycle(alignment.cycle_id).regime_pending.pop(alignment.cycle_id, None)
        return record

    def _state_of_cycle(self, cycle_id: str) -> MachineState:
        for state in self._machines.values():
            if cycle_id in state.regime_pending:
                return state
        cycle = self.store.get_cycle(cycle_id)
        return self._state(cycle.machine_id if cycle is not None else self.default_machine)

    def _track_good(self, state: MachineState, alignment: AlignmentResult, camera_id: str, verdict: str,
                    source: str) -> None:
        """A cycle joins the regime's good set once every required view of every part passed it."""
        cycle_id = alignment.cycle_id
        if cycle_id not in state.regime_pending:
            return
        if verdict != PASS or source != "vision":
            state.regime_pending.pop(cycle_id)
            self._views_ok.pop(cycle_id, None)
            return
        seen = self._views_ok.setdefault(cycle_id, set())
        seen.add((camera_id, alignment.cavity))
        while len(self._views_ok) > 512:
            self._views_ok.popitem(last=False)
        link = self.aligner.link_for(camera_id)
        cavities = range(link.parts_per_cycle if link is not None else 1)
        wanted = {(c, k) for c in (self.required or (camera_id,)) for k in cavities}
        if wanted <= seen:
            state.regime_good.append(state.regime_pending.pop(cycle_id))
            self._views_ok.pop(cycle_id, None)

    def record_audit(self, cycle_id: str, verdict: str, *, defect_type: str | None = None,
                     auditor: str = "audit") -> InspectionRecord:
        """Ground truth from manual re-inspection or customer returns (overrides vision)."""
        return self.store.save_inspection(InspectionRecord(
            inspection_id=uuid.uuid4().hex, camera_id=auditor, ts=time.time(), verdict=verdict,
            defect_type=defect_type if verdict == FAIL else None, cycle_id=cycle_id,
            alignment_confidence=1.0, source="audit",
        ))

    def telemetry_for(self, cycle_id: str | None) -> dict[str, Any]:
        """Telemetry of the producing cycle, shaped like TelemetryContext fields."""
        if not cycle_id:
            return {}
        cycle = self.store.get_cycle(cycle_id)
        features = self.store.get_features(cycle_id) or {}
        if cycle is None:
            return {}
        out: dict[str, Any] = {"machine_id": cycle.machine_id, "line_id": cycle.line_id,
                               "lot_id": cycle.scalars.get("material_lot")}
        for field_name, keys in _TELEMETRY_FEATURES.items():
            for key in keys:
                if key in features:
                    out[field_name] = float(features[key])
                    break
        return out

    # ------------------------------------------------------------- training

    def retrain_async(self, machine_id: str | None = None) -> None:
        if self._retrain_lock.locked():
            return
        threading.Thread(target=self.retrain, args=(machine_id,), name="process-retrain", daemon=True).start()

    def retrain(self, machine_id: str | None = None) -> dict[str, Any]:
        """Refresh one machine's golden reference and risk model from its verified labels."""
        if not self._retrain_lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
            with self._lock:
                state = self._state(machine_id or self.default_machine)
                state.labels_since_train = 0
                since = state.reset_ts

            good_cycles = self.store.good_cycles(limit=max(self.golden_warmup_cycles, 400), with_curves=True,
                                                 machine_id=state.machine_id, since_ts=since,
                                                 min_views=max(1, len(self.required)))
            golden = GoldenCycleReference.fit(good_cycles, min_cycles=self.golden_warmup_cycles)
            extractor = state.extractor
            if golden is not None:
                extractor = CycleFeatureExtractor(golden)

            # Only the newest cycles: the cost stays flat however long the line has run.
            rows = self.store.labelled_dataset(self.max_train_cycles, min_confidence=self.min_label_confidence,
                                               machine_id=state.machine_id, since_ts=since,
                                               required_cameras=self.required)
            if golden is not None:
                # Re-extract with the verified golden so features match what will be served.
                for row in rows:
                    cycle = self.store.get_cycle(row["cycle_id"], with_curves=True)
                    if cycle is not None and cycle.curves:
                        row["features"] = extractor.extract(cycle)

            n_pos = sum(r["label"] for r in rows)
            report: dict[str, Any] = {"labelled": len(rows), "positives": n_pos}
            candidate: DefectRiskModel | None = None
            try:
                candidate = train_supervised_model(
                    rows, min_samples=self.min_train_samples, min_positives=self.min_train_positives,
                    spec_limits=self.spec_limits, hybrid_deviation=self.hybrid_deviation,
                )
                report["candidate"] = candidate.metrics
            except ValueError as e:
                report["supervised_skipped"] = str(e)
                good_rows = [r for r in rows if not r["label"]]
                if good_rows:
                    try:
                        candidate = fit_deviation_model(good_rows)
                    except ValueError as e2:
                        report["deviation_skipped"] = str(e2)

            if candidate is not None:
                # Judge the candidate exactly as it will serve.
                candidate.spec_limits = self.spec_limits
                candidate.hybrid_deviation = self.hybrid_deviation
            with self._lock:
                incumbent = state.model
            incumbent_auc = None
            if candidate is not None and candidate.kind == "supervised" and incumbent is not None:
                # Score the incumbent on exactly the candidate's held-out window.
                incumbent_auc = evaluate_auc(incumbent, validation_rows(rows))
                report["incumbent_auc_same_window"] = incumbent_auc
            promote = candidate is not None and is_better(
                candidate, incumbent, min_auc=self.min_promotion_auc, incumbent_auc=incumbent_auc,
            )
            # A deviation model is always refreshed when no supervised model is active.
            if candidate is not None and candidate.kind == "deviation" and (incumbent is None or incumbent.kind == "deviation"):
                promote = True
            with self._lock:
                if promote and state.reset_ts != since:
                    promote = False             # the process changed while this was training
                    report["discarded"] = "the process changed during training"
                if promote:
                    if golden is not None:
                        state.golden, state.extractor = golden, extractor
                        self._save_golden(state.machine_id, golden)
                    self._activate(state, candidate, n_samples=len(rows), n_positive=n_pos)
            report["promoted"] = bool(promote)
            report["active_version"] = state.model.version if state.model else None
            logger.info("[Process] Retrain report: %s", report)
            return report
        finally:
            self._retrain_lock.release()

    def _activate(self, state: MachineState, model: DefectRiskModel, *, n_samples: int, n_positive: int) -> None:
        model.spec_limits = self.spec_limits
        model.hybrid_deviation = self.hybrid_deviation
        path = model.save(self.models_dir)
        self.store.register_model(model.version, model.kind, path, n_samples, n_positive, model.metrics, activate=True,
                                  machine_id=state.machine_id)
        with self._lock:
            state.model = model
            state.regime_reference, state.regime_reference_size = None, -1
        logger.info("[Process] Activated %s risk model %s", model.kind, model.version)

    # ---------------------------------------------------------------- KPIs

    def kpis(self, window_s: float = 8 * 3600.0) -> dict[str, Any]:
        now = time.time()
        report = compute_kpis(self.store, now - window_s, now, required_cameras=self.required)
        report["model"] = (
            {"version": self.model.version, "kind": self.model.kind, "metrics": self.model.metrics}
            if self.model else None
        )
        return report
