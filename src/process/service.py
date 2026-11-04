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
from pathlib import Path
from typing import Any, Callable

from src.process.alignment import AlignmentResult, CycleAligner, StationLink
from src.process.features import CycleFeatureExtractor, GoldenCycleReference
from src.process.kpis import compute_kpis
from src.process.models import FAIL, PASS, InspectionRecord, ProcessCycle, RiskAssessment
from src.process.prescriptions import Prescription, PrescriptionEngine
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
from src.process.store import ProcessStore

logger = logging.getLogger("QualyxusProcessIntelligence")

PRUNE_EVERY_CYCLES = 500

AlertCallback = Callable[[RiskAssessment, ProcessCycle, list[Prescription]], None]

# Telemetry fields carried on DefectEvent records, resolved from the producing cycle.
_TELEMETRY_FEATURES = {
    "temperature": ("melt_temp.mean", "melt_temp"),
    "speed": ("setpoint.injection_speed_sp", "injection_speed"),
}


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
        self._labels_since_train = 0
        self._provisional_good: list[ProcessCycle] = []
        self.source = None

        # Operating regime = the active setpoint combination. Intentional setpoint changes
        # move measured signals by design; deviation scoring is re-baselined per regime.
        self._regime_setpoints: dict[str, float] | None = None
        self._regime_changed_groups: frozenset[str] = frozenset()
        self._regime_pending: "collections.OrderedDict[str, dict[str, float]]" = collections.OrderedDict()
        self._regime_good: list[dict[str, float]] = []
        self._regime_reference: DeviationReference | None = None
        self._regime_reference_size = -1

        self.golden = self._load_golden()
        self.extractor = CycleFeatureExtractor(self.golden)
        self.model = self._load_active_model()
        if self.model is not None:
            self.model.spec_limits = self.spec_limits
            self.model.hybrid_deviation = self.hybrid_deviation

    # ----------------------------------------------------------- persistence

    @property
    def _golden_path(self) -> Path:
        return self.models_dir / "golden_cycle.json"

    def _load_golden(self) -> GoldenCycleReference | None:
        try:
            if self._golden_path.exists():
                return GoldenCycleReference.from_dict(json.loads(self._golden_path.read_text(encoding="utf-8")))
        except Exception as e:
            logger.warning("[Process] Ignoring unreadable golden reference: %s", e)
        return None

    def _save_golden(self, golden: GoldenCycleReference) -> None:
        tmp = self._golden_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(golden.to_dict()), encoding="utf-8")
        tmp.replace(self._golden_path)

    def _load_active_model(self) -> DefectRiskModel | None:
        entry = self.store.active_model()
        if not entry:
            return None
        try:
            model = DefectRiskModel.load(entry["path"])
            logger.info("[Process] Loaded active %s risk model %s", model.kind, model.version)
            return model
        except Exception as e:
            logger.warning("[Process] Active model %s could not be loaded: %s", entry.get("version"), e)
            return None

    @property
    def tracker(self):
        if self._tracker is None:
            from src.operations.action_tracker import ActionTracker

            self._tracker = ActionTracker(
                sqlite_path=str(Path(self.store.db_path).parent / "actions.db")
            )
        return self._tracker

    # -------------------------------------------------------------- wiring

    def add_alert_callback(self, callback: AlertCallback) -> None:
        self._alert_callbacks.append(callback)

    def attach_source(self, source) -> None:
        self.source = source
        source.subscribe(self.on_cycle)

    def start(self) -> None:
        if self.source is not None:
            self.source.start()

    def stop(self) -> None:
        if self.source is not None:
            self.source.stop()

    # ------------------------------------------------------------ ingestion

    def on_cycle(self, cycle: ProcessCycle) -> RiskAssessment | None:
        with self._lock:
            if self.golden is None:
                self._bootstrap_golden(cycle)
            features = self.extractor.extract(cycle)
            self.store.save_cycle(cycle, features)
            self._cycles_seen += 1
            if self.curve_retention_s > 0 and self._cycles_seen % PRUNE_EVERY_CYCLES == 0:
                self.store.prune_curves(self.curve_retention_s)
            model = self.model
            self._track_regime(cycle, features)
            reference = self._current_reference(model)

        if model is None:
            return None
        assessment = model.assess(cycle.cycle_id, features, reference=reference)
        self.store.save_prediction(assessment)
        prescriptions: list[Prescription] = []
        if assessment.decision == "high" and self.prescriptions_enabled:
            prescriptions = self.prescriber.recommend(
                assessment, features, cycle.setpoints, model,
                machine_id=cycle.machine_id, sequence=cycle.sequence, reference=reference,
            )
            for presc in prescriptions:
                try:
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

    def _track_regime(self, cycle: ProcessCycle, features: dict[str, float]) -> None:
        setpoints = {k: float(v) for k, v in (cycle.setpoints or {}).items()}
        if self._regime_setpoints is None:
            self._regime_setpoints = setpoints
        elif setpoints != self._regime_setpoints:
            changed = {k for k in set(setpoints) | set(self._regime_setpoints)
                       if abs(setpoints.get(k, float("nan")) - self._regime_setpoints.get(k, float("nan"))) > 1e-9
                       or (k in setpoints) != (k in self._regime_setpoints)}
            groups = set(self._regime_changed_groups)
            for driver, rule in self.knowledge.items():
                if rule.get("setpoint") in changed:
                    groups.update(rule.get("effects") or [DefectRiskModel.signal_group(driver)])
            logger.info("[Process] Operating regime changed (%s); re-baselining %s",
                        ", ".join(sorted(changed)), ", ".join(sorted(groups)) or "nothing")
            self._regime_setpoints = setpoints
            self._regime_changed_groups = frozenset(groups)
            self._regime_pending.clear()
            self._regime_good = []
            self._regime_reference, self._regime_reference_size = None, -1
        self._regime_pending[cycle.cycle_id] = features
        while len(self._regime_pending) > 500:
            self._regime_pending.popitem(last=False)

    def _current_reference(self, model: DefectRiskModel | None) -> DeviationReference | None:
        if model is None:
            return None
        if len(self._regime_good) >= self.min_regime_good:
            # Refresh the regime baseline as verified-good cycles accumulate.
            if self._regime_reference is None or len(self._regime_good) - self._regime_reference_size >= 10:
                self._regime_reference = model.reference_from_good(self._regime_good[-500:])
                self._regime_reference_size = len(self._regime_good)
            return self._regime_reference
        if self._regime_changed_groups:
            return DeviationReference(exclude_groups=self._regime_changed_groups)
        return None

    def _bootstrap_golden(self, cycle: ProcessCycle) -> None:
        """Before any inspection exists, assume the first cycles of a run are healthy.

        The provisional reference is replaced by one built from *verified* good cycles on
        the first retrain, so a process that starts unhealthy self-corrects.
        """
        self._provisional_good.append(cycle)
        if len(self._provisional_good) < self.golden_warmup_cycles:
            return
        golden = GoldenCycleReference.fit(self._provisional_good, min_cycles=self.golden_warmup_cycles)
        if golden is None:
            return
        self.golden = golden
        self.extractor = CycleFeatureExtractor(golden)
        self._save_golden(golden)
        rows = [{"end_ts": c.end_ts, "features": self.extractor.extract(c)} for c in self._provisional_good]
        try:
            self._activate(fit_deviation_model(rows, min_rows=min(20, len(rows))), n_samples=len(rows), n_positive=0)
        except ValueError as e:
            logger.warning("[Process] Deviation bootstrap skipped: %s", e)
        self._provisional_good = []
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
        if verdict not in (PASS, FAIL):
            raise ValueError(f"verdict must be '{PASS}' or '{FAIL}', got {verdict!r}")
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
        if alignment.cycle_id:
            with self._lock:
                self._labels_since_train += 1
                features = self._regime_pending.pop(alignment.cycle_id, None)
                if features is not None and verdict == PASS and source == "vision":
                    self._regime_good.append(features)
                due = self._labels_since_train >= self.retrain_every
            if due:
                self.retrain_async()
        return record

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

    def retrain_async(self) -> None:
        if self._retrain_lock.locked():
            return
        threading.Thread(target=self.retrain, name="process-retrain", daemon=True).start()

    def retrain(self) -> dict[str, Any]:
        """Refresh the golden reference and risk model from verified labels."""
        if not self._retrain_lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
            with self._lock:
                self._labels_since_train = 0

            good_cycles = self.store.good_cycles(limit=max(self.golden_warmup_cycles, 400), with_curves=True)
            golden = GoldenCycleReference.fit(good_cycles, min_cycles=self.golden_warmup_cycles)
            extractor = self.extractor
            if golden is not None:
                extractor = CycleFeatureExtractor(golden)

            rows = self.store.labelled_dataset(min_confidence=self.min_label_confidence)
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
                incumbent = self.model
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
            if promote:
                with self._lock:
                    if golden is not None:
                        self.golden, self.extractor = golden, extractor
                        self._save_golden(golden)
                self._activate(candidate, n_samples=len(rows), n_positive=n_pos)
            report["promoted"] = bool(promote)
            report["active_version"] = self.model.version if self.model else None
            logger.info("[Process] Retrain report: %s", report)
            return report
        finally:
            self._retrain_lock.release()

    def _activate(self, model: DefectRiskModel, *, n_samples: int, n_positive: int) -> None:
        model.spec_limits = self.spec_limits
        model.hybrid_deviation = self.hybrid_deviation
        path = model.save(self.models_dir)
        self.store.register_model(model.version, model.kind, path, n_samples, n_positive, model.metrics, activate=True)
        with self._lock:
            self.model = model
            self._regime_reference, self._regime_reference_size = None, -1
        logger.info("[Process] Activated %s risk model %s", model.kind, model.version)

    # ---------------------------------------------------------------- KPIs

    def kpis(self, window_s: float = 8 * 3600.0) -> dict[str, Any]:
        now = time.time()
        report = compute_kpis(self.store, now - window_s, now)
        report["model"] = (
            {"version": self.model.version, "kind": self.model.kind, "metrics": self.model.metrics}
            if self.model else None
        )
        return report
