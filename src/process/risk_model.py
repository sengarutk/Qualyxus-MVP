"""Per-cycle defect-risk models.

Two model kinds share one interface:

``deviation``   Unsupervised. Robust z-scores of every measured feature against the
                distribution of known-good cycles. Works from day one with no defect
                labels at all, and stays as the fallback.
``supervised``  Gradient-boosted trees trained on cycles labelled by vision inspections
                (and audits). Trained with a time-ordered validation split - never a
                random split, which would leak future process state into the past - and
                only activated when it beats chance on held-out data.

Every prediction carries counterfactual explanations: for each feature, how much the risk
would drop if that feature alone returned to its good-cycle median. Those factors drive
the prescription engine.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.inference import signed_model
from src.process.models import RiskAssessment

logger = logging.getLogger("QualyxusRiskModel")

# Setpoints are deliberate operator choices; deviating from history is not a symptom.
_EXCLUDED_FROM_DEVIATION = ("setpoint.",)


def _robust_stats(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(matrix, axis=0)
    mad = np.median(np.abs(matrix - median), axis=0) * 1.4826
    # Floor the spread so near-constant features do not explode the z-score.
    floor = np.maximum(1e-6, 0.01 * np.abs(median))
    return median, np.maximum(mad, floor)


@dataclass
class DeviationReference:
    """Override of what "normal" means for the deviation term.

    After an intentional setpoint change the process legitimately moves: ``exclude_groups``
    removes the signal groups causally downstream of the changed setpoints until a new
    baseline (``median``/``scale`` from verified-good cycles of the new regime) exists.
    """

    median: np.ndarray | None = None
    scale: np.ndarray | None = None
    exclude_groups: frozenset[str] = frozenset()


@dataclass
class DefectRiskModel:
    kind: str
    version: str
    feature_names: list[str]
    good_median: np.ndarray
    good_scale: np.ndarray
    watch_threshold: float
    high_threshold: float
    metrics: dict[str, Any] = field(default_factory=dict)
    classifier: Any = None
    type_classifier: Any = None
    type_labels: list[str] = field(default_factory=list)
    deviation_mask: np.ndarray | None = None
    # Engineering specification limits (material datasheets, process windows). Applied at
    # serving time and not learned, so they are set by the service from configuration.
    spec_limits: dict[str, dict[str, float]] = field(default_factory=dict)
    # Whether the supervised model also ORs in the unsupervised deviation term. Off by
    # default: intentional setpoint changes move the process into regimes the historical
    # "normal" never covered, so the term floods with false alarms once the loop is closed.
    hybrid_deviation: bool = False

    def spec_risk(self, matrix: np.ndarray) -> np.ndarray:
        """Probability-like risk from specification-limit violations (0 inside all windows)."""
        survival = np.ones(matrix.shape[0])
        for feature, spec in (getattr(self, "spec_limits", None) or {}).items():
            if feature not in self.feature_names:
                continue
            col = matrix[:, self.feature_names.index(feature)]
            violation = np.zeros_like(col)
            if spec.get("min") is not None:
                violation = np.maximum(violation, float(spec["min"]) - col)
            if spec.get("max") is not None:
                violation = np.maximum(violation, col - float(spec["max"]))
            softness = max(float(spec.get("softness", 1.0)), 1e-6)
            survival *= np.exp(-np.maximum(violation, 0.0) / softness)
        return 1.0 - survival

    # ------------------------------------------------------------- inference

    def vectorize(self, features: dict[str, float]) -> np.ndarray:
        row = np.array([float(features.get(name, np.nan)) for name in self.feature_names], dtype=np.float64)
        # Missing features fall back to the good median so they carry no risk signal.
        missing = ~np.isfinite(row)
        row[missing] = self.good_median[missing]
        return row

    def _z(self, matrix: np.ndarray, reference: DeviationReference | None = None) -> np.ndarray:
        median = reference.median if reference is not None and reference.median is not None else self.good_median
        scale = reference.scale if reference is not None and reference.scale is not None else self.good_scale
        return (matrix - median) / scale

    def _deviation_mask(self, reference: DeviationReference | None) -> np.ndarray:
        mask = self.deviation_mask if self.deviation_mask is not None else np.ones(len(self.feature_names), bool)
        if reference is not None and reference.exclude_groups:
            keep = np.array([self.signal_group(n) not in reference.exclude_groups for n in self.feature_names])
            mask = mask & keep
        return mask

    def deviation_risk(self, matrix: np.ndarray, reference: DeviationReference | None = None) -> np.ndarray:
        z = np.abs(self._z(matrix, reference))
        mask = self._deviation_mask(reference)
        if not mask.any():
            return np.zeros(matrix.shape[0])
        z = z[:, mask]
        # Mean of the three largest deviations: robust to a single noisy channel while
        # still reacting to one strongly abnormal signal.
        k = min(3, z.shape[1])
        top = np.sort(z, axis=1)[:, -k:].mean(axis=1)
        return 1.0 / (1.0 + np.exp(-(top - 4.0) / 0.8))

    def risk_matrix(self, matrix: np.ndarray, reference: DeviationReference | None = None) -> np.ndarray:
        # Noisy-OR of independent evidence: the classifier knows the failure modes it has
        # seen, the deviation term fires on novel operating regions, and specification
        # limits encode engineering knowledge no amount of history may contain.
        survival = 1.0 - self.spec_risk(matrix)
        if self.kind == "supervised" and self.classifier is not None:
            survival *= 1.0 - self.classifier.predict_proba(matrix)[:, 1]
            if getattr(self, "hybrid_deviation", False):
                survival *= 1.0 - self.deviation_risk(matrix, reference)
        else:
            survival *= 1.0 - self.deviation_risk(matrix, reference)
        return 1.0 - survival

    def assess(self, cycle_id: str, features: dict[str, float], top_k: int = 5,
               reference: DeviationReference | None = None) -> RiskAssessment:
        row = self.vectorize(features)
        risk = float(self.risk_matrix(row[None, :], reference)[0])
        decision = "high" if risk >= self.high_threshold else ("watch" if risk >= self.watch_threshold else "ok")
        factors = self.explain(row, risk, top_k=top_k, reference=reference) if decision != "ok" else []
        likely, likely_p = self._likely_defect(row) if decision != "ok" else (None, None)
        return RiskAssessment(
            cycle_id=cycle_id,
            risk=round(risk, 5),
            decision=decision,
            model_version=self.version,
            model_kind=self.kind,
            likely_defect=likely,
            likely_defect_probability=likely_p,
            top_factors=factors,
            ts=time.time(),
        )

    @staticmethod
    def signal_group(feature: str) -> str:
        """Features derived from the same sensor share a group (e.g. ``melt_temp.*``)."""
        return feature.split(".", 1)[0] if "." in feature and not feature.startswith("setpoint.") else feature

    def group_members(self, feature: str) -> list[int]:
        group = self.signal_group(feature)
        return [i for i, name in enumerate(self.feature_names) if self.signal_group(name) == group]

    def explain(self, row: np.ndarray, risk: float | None = None, top_k: int = 5,
                reference: DeviationReference | None = None) -> list[dict[str, Any]]:
        """Counterfactual attribution per signal group.

        One abnormal sensor moves many correlated features (mean, plateau, peak, golden
        deviation...). Resetting a single feature leaves its siblings abnormal and hides the
        cause, so each *group* is returned to its good-cycle median together; the group's
        most deviant feature is reported as the representative.
        """
        if risk is None:
            risk = float(self.risk_matrix(row[None, :], reference)[0])
        z = self._z(row[None, :], reference)[0]
        normal = reference.median if reference is not None and reference.median is not None else self.good_median
        groups: dict[str, list[int]] = {}
        for idx, name in enumerate(self.feature_names):
            groups.setdefault(self.signal_group(name), []).append(idx)
        names = list(groups)
        counterfactuals = np.repeat(row[None, :], len(names), axis=0)
        for k, group in enumerate(names):
            members = groups[group]
            counterfactuals[k, members] = normal[members]
        drops = risk - self.risk_matrix(counterfactuals, reference)

        candidates = []
        for k, group in enumerate(names):
            members = groups[group]
            rep = max(members, key=lambda i: abs(z[i]))
            candidates.append((float(drops[k]), abs(float(z[rep])), rep, group))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

        out = []
        for drop, abs_z, idx, group in candidates:
            if drop <= 1e-4 and abs_z < 3.0:
                continue
            out.append({
                "feature": self.feature_names[idx],
                "group": group,
                "value": round(float(row[idx]), 5),
                "normal": round(float(normal[idx]), 5),
                "z": round(float(z[idx]), 3),
                "direction": "above" if row[idx] > normal[idx] else "below",
                "risk_drop_if_normal": round(float(max(drop, 0.0)), 5),
            })
            if len(out) >= top_k:
                break
        return out

    def feature_z(self, features: dict[str, float], feature: str,
                  reference: DeviationReference | None = None) -> float | None:
        if feature not in self.feature_names:
            return None
        idx = self.feature_names.index(feature)
        row = self.vectorize(features)[None, :]
        return float(self._z(row, reference)[0, idx])

    def normal_overrides_for_groups(self, groups: list[str],
                                    reference: DeviationReference | None = None) -> dict[str, float]:
        wanted = set(groups)
        normal = reference.median if reference is not None and reference.median is not None else self.good_median
        return {name: float(normal[i]) for i, name in enumerate(self.feature_names)
                if self.signal_group(name) in wanted}

    def normal_group_overrides(self, feature: str) -> dict[str, float]:
        """Good-cycle medians for every feature in ``feature``'s signal group."""
        return {self.feature_names[i]: float(self.good_median[i]) for i in self.group_members(feature)}

    def _likely_defect(self, row: np.ndarray) -> tuple[str | None, float | None]:
        if self.type_classifier is None or not self.type_labels:
            return None, None
        probs = self.type_classifier.predict_proba(row[None, :])[0]
        idx = int(np.argmax(probs))
        return str(self.type_classifier.classes_[idx]), round(float(probs[idx]), 4)

    def counterfactual_risk(self, features: dict[str, float], overrides: dict[str, float],
                            reference: DeviationReference | None = None) -> float:
        row = self.vectorize({**features, **overrides})
        return float(self.risk_matrix(row[None, :], reference)[0])

    def reference_from_good(self, good_feature_rows: Sequence[dict[str, float]],
                            exclude_groups: frozenset[str] = frozenset()) -> DeviationReference:
        """Baseline re-learned from verified-good cycles of the current operating regime."""
        matrix = np.array([self.vectorize(r) for r in good_feature_rows])
        median, scale = _robust_stats(matrix)
        # Never claim more certainty than the model-wide good distribution provides.
        return DeviationReference(median=median, scale=np.maximum(scale, self.good_scale), exclude_groups=exclude_groups)

    # ---------------------------------------------------------- persistence

    def save(self, directory: str | Path) -> str:
        path = Path(directory) / f"risk_{self.version}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        signed_model.save(self, path)
        return str(path)

    @staticmethod
    def load(path: str | Path) -> "DefectRiskModel":
        model = signed_model.load(path)
        if not isinstance(model, DefectRiskModel):
            raise TypeError(f"{path} does not contain a DefectRiskModel")
        return model


def _new_version(kind: str) -> str:
    return f"{kind}-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _feature_names(rows: Sequence[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(row["features"].keys())
    return sorted(names)


def _matrix(rows: Sequence[dict[str, Any]], names: list[str]) -> np.ndarray:
    matrix = np.array([[float(r["features"].get(n, np.nan)) for n in names] for r in rows], dtype=np.float64)
    if matrix.size:
        col_median = np.nanmedian(np.where(np.isfinite(matrix), matrix, np.nan), axis=0)
        col_median = np.where(np.isfinite(col_median), col_median, 0.0)
        bad = ~np.isfinite(matrix)
        matrix[bad] = np.take(col_median, np.where(bad)[1])
    return matrix


def fit_deviation_model(good_rows: Sequence[dict[str, Any]], *, min_rows: int = 20,
                        max_rows: int = 1500) -> DefectRiskModel:
    if len(good_rows) < min_rows:
        raise ValueError(f"Deviation model needs >= {min_rows} good cycles (got {len(good_rows)})")
    good_rows = sorted(good_rows, key=lambda r: r.get("end_ts", 0.0))[-max_rows:]
    names = _feature_names(good_rows)
    matrix = _matrix(good_rows, names)
    median, scale = _robust_stats(matrix)
    mask = np.array([not n.startswith(_EXCLUDED_FROM_DEVIATION) for n in names])
    model = DefectRiskModel(
        kind="deviation",
        version=_new_version("deviation"),
        feature_names=names,
        good_median=median,
        good_scale=scale,
        watch_threshold=0.35,
        high_threshold=0.7,
        deviation_mask=mask,
        metrics={"n_good": len(good_rows)},
    )
    return model


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    metrics = {
        "n": int(len(y)),
        "base_rate": round(float(np.mean(y)), 5),
        "precision": round(tp / (tp + fp), 5) if (tp + fp) else 0.0,
        "recall": round(tp / (tp + fn), 5) if (tp + fn) else 0.0,
        "brier": round(float(brier_score_loss(y, p)), 5),
    }
    if len(np.unique(y)) == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y, p)), 5)
        metrics["average_precision"] = round(float(average_precision_score(y, p)), 5)
    return metrics


def _best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.1, 0.9, 33):
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        fn = np.sum(~pred & (y == 1))
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t


def _make_classifier(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=seed,
    )


def train_supervised_model(
    rows: Sequence[dict[str, Any]],
    *,
    min_samples: int = 300,
    min_positives: int = 20,
    val_fraction: float = 0.25,
    good_reference_cycles: int = 1500,
    seed: int = 42,
    spec_limits: dict[str, dict[str, float]] | None = None,
    hybrid_deviation: bool = False,
) -> DefectRiskModel:
    """Train on time-ordered labelled cycles. Raises ValueError when data is insufficient."""
    rows = sorted(rows, key=lambda r: r["end_ts"])
    y_all = np.array([int(r["label"]) for r in rows])
    if len(rows) < min_samples or int(y_all.sum()) < min_positives:
        raise ValueError(
            f"Need >= {min_samples} labelled cycles and >= {min_positives} defects "
            f"(have {len(rows)} / {int(y_all.sum())})"
        )
    names = _feature_names(rows)
    X_all = _matrix(rows, names)
    mask = np.array([not n.startswith(_EXCLUDED_FROM_DEVIATION) for n in names])

    def _good_stats(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        good = X[y == 0][-good_reference_cycles:]
        return _robust_stats(good if len(good) >= 10 else X)

    split = int(len(rows) * (1.0 - val_fraction))
    X_tr, y_tr, X_va, y_va = X_all[:split], y_all[:split], X_all[split:], y_all[split:]
    if len(np.unique(y_tr)) < 2:
        raise ValueError("Training window contains a single class; collect more varied data")

    # Held-out estimate of the *deployed* hybrid scorer, built only from the training window.
    median_tr, scale_tr = _good_stats(X_tr, y_tr)
    probe = DefectRiskModel(
        kind="supervised", version="probe", feature_names=names, good_median=median_tr,
        good_scale=scale_tr, watch_threshold=0.0, high_threshold=1.0,
        classifier=_make_classifier(seed).fit(X_tr, y_tr), deviation_mask=mask,
        spec_limits=dict(spec_limits or {}), hybrid_deviation=hybrid_deviation,
    )
    p_va = probe.risk_matrix(X_va)
    threshold = _best_f1_threshold(y_va, p_va) if len(np.unique(y_va)) == 2 else 0.5
    validation = _binary_metrics(y_va, p_va, threshold)

    # Final model uses all data; thresholds come from the honest held-out estimate.
    final = _make_classifier(seed).fit(X_all, y_all)
    median, scale = _good_stats(X_all, y_all)

    type_clf, type_labels = None, []
    fail_rows = [r for r in rows if r["label"] and r.get("defect_type")]
    labels = [r["defect_type"] for r in fail_rows]
    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    keep = [i for i, lbl in enumerate(labels) if counts[lbl] >= 5]
    if len({labels[i] for i in keep}) >= 2:
        from sklearn.ensemble import HistGradientBoostingClassifier

        X_fail = _matrix([fail_rows[i] for i in keep], names)
        type_clf = HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15, random_state=seed)
        type_clf.fit(X_fail, [labels[i] for i in keep])
        type_labels = sorted({labels[i] for i in keep})

    return DefectRiskModel(
        kind="supervised",
        version=_new_version("supervised"),
        feature_names=names,
        good_median=median,
        good_scale=scale,
        watch_threshold=round(max(0.05, threshold * 0.5), 4),
        high_threshold=round(threshold, 4),
        metrics={"validation": validation, "n_train": int(split), "n_total": int(len(rows)),
                 "n_positive": int(y_all.sum())},
        classifier=final,
        type_classifier=type_clf,
        type_labels=type_labels,
        deviation_mask=mask,
        spec_limits=dict(spec_limits or {}),
        hybrid_deviation=hybrid_deviation,
    )


def evaluate_auc(model: DefectRiskModel, rows: Sequence[dict[str, Any]]) -> float | None:
    """ROC AUC of ``model`` on labelled rows (None when only one class is present)."""
    y = np.array([int(r["label"]) for r in rows])
    if len(rows) == 0 or len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    X = np.array([model.vectorize(r["features"]) for r in rows])
    return float(roc_auc_score(y, model.risk_matrix(X)))


def validation_rows(rows: Sequence[dict[str, Any]], val_fraction: float = 0.25) -> list[dict[str, Any]]:
    """The most recent ``val_fraction`` of time-ordered rows (the candidate's held-out window)."""
    ordered = sorted(rows, key=lambda r: r["end_ts"])
    return ordered[int(len(ordered) * (1.0 - val_fraction)):]


def is_better(candidate: DefectRiskModel, incumbent: DefectRiskModel | None, *, min_auc: float = 0.65,
              tolerance: float = 0.02, incumbent_auc: float | None = None) -> bool:
    """Promotion rule: a supervised model must beat chance clearly and not regress.

    ``incumbent_auc`` must be measured on the *same* held-out window as the candidate's
    validation AUC; comparing metrics from different windows is meaningless.
    """
    if candidate.kind != "supervised":
        return incumbent is None or incumbent.kind == "deviation"
    cand_auc = candidate.metrics.get("validation", {}).get("roc_auc")
    if cand_auc is None or not math.isfinite(cand_auc) or cand_auc < min_auc:
        return False
    if incumbent is None or incumbent_auc is None:
        return True
    return cand_auc >= incumbent_auc - tolerance
