"""Business KPIs that a quality buyer actually pays for.

* scrap rate                 - share of inspected parts that failed
* prediction quality         - how well per-cycle risk separated failing from good cycles
* early-warning lead time    - how far ahead of a defect episode the risk was raised
* false reject / escape rate - only computable against audits (manual re-inspection,
                               customer returns); reported as ``None`` when absent so a
                               missing ground truth is never mistaken for a perfect score.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.process.models import FAIL
from src.process.store import ProcessStore


def compute_kpis(
    store: ProcessStore,
    ts_from: float,
    ts_to: float,
    *,
    min_quiet_parts: int = 10,
) -> dict[str, Any]:
    inspections = [dict(r) for r in store.inspections_between(ts_from, ts_to)]
    vision = [r for r in inspections if r["source"] == "vision"]
    audits = [r for r in inspections if r["source"] == "audit"]

    n_parts = len(vision)
    n_fail = sum(1 for r in vision if r["verdict"] == FAIL)
    linked = [r for r in vision if r["cycle_id"]]

    kpis: dict[str, Any] = {
        "window": {"from": ts_from, "to": ts_to},
        "parts_inspected": n_parts,
        "parts_failed": n_fail,
        "scrap_rate": round(n_fail / n_parts, 5) if n_parts else None,
        "alignment_coverage": round(len(linked) / n_parts, 5) if n_parts else None,
        "mean_alignment_confidence": (
            round(float(np.mean([r["alignment_confidence"] or 0.0 for r in linked])), 4) if linked else None
        ),
    }

    # --- audits: false rejects & escapes ------------------------------------------
    vision_by_cycle: dict[str, str] = {}
    for r in linked:
        if r["verdict"] == FAIL or r["cycle_id"] not in vision_by_cycle:
            vision_by_cycle[r["cycle_id"]] = r["verdict"]
    audited = [(vision_by_cycle.get(a["cycle_id"]), a["verdict"]) for a in audits if a["cycle_id"] in vision_by_cycle]
    vision_fail_audited = [a for v, a in audited if v == FAIL]
    vision_pass_audited = [a for v, a in audited if v != FAIL]
    kpis["audits"] = len(audits)
    kpis["false_reject_rate"] = (
        round(sum(1 for a in vision_fail_audited if a != FAIL) / len(vision_fail_audited), 5)
        if vision_fail_audited else None
    )
    kpis["escape_rate"] = (
        round(sum(1 for a in vision_pass_audited if a == FAIL) / len(vision_pass_audited), 5)
        if vision_pass_audited else None
    )

    # --- prediction quality on cycles with both a prediction and a vision label -------
    predictions = {r["cycle_id"]: dict(r) for r in store.predictions_between(ts_from - 86400.0, ts_to)}
    labelled = [(predictions[c], 1 if v == FAIL else 0) for c, v in vision_by_cycle.items() if c in predictions]
    if labelled:
        risks = np.array([p["risk"] for p, _ in labelled])
        y = np.array([lbl for _, lbl in labelled])
        flagged = np.array([p["decision"] == "high" for p, _ in labelled])
        tp = int(np.sum(flagged & (y == 1)))
        fp = int(np.sum(flagged & (y == 0)))
        fn = int(np.sum(~flagged & (y == 1)))
        quality: dict[str, Any] = {
            "cycles_scored": int(len(y)),
            "precision": round(tp / (tp + fp), 5) if (tp + fp) else None,
            "recall": round(tp / (tp + fn), 5) if (tp + fn) else None,
        }
        if len(np.unique(y)) == 2:
            from sklearn.metrics import average_precision_score, roc_auc_score

            quality["roc_auc"] = round(float(roc_auc_score(y, risks)), 5)
            quality["average_precision"] = round(float(average_precision_score(y, risks)), 5)
        kpis["prediction"] = quality
    else:
        kpis["prediction"] = None

    # --- early warning lead time ----------------------------------------------------
    # For every defect episode (a failure after >= min_quiet_parts good parts), count how
    # many consecutive cycles immediately before it were already flagged (watch/high).
    ordered = sorted(linked, key=lambda r: r["ts"])
    leads_cycles: list[int] = []
    leads_seconds: list[float] = []
    quiet = 0
    for idx, r in enumerate(ordered):
        if r["verdict"] != FAIL:
            quiet += 1
            continue
        if quiet >= min_quiet_parts:
            lead = 0
            first_flag_ts = None
            for prev in reversed(ordered[:idx + 1]):
                pred = predictions.get(prev["cycle_id"])
                if not pred or pred["decision"] == "ok":
                    break
                lead += 1
                first_flag_ts = pred["ts"]
            leads_cycles.append(max(0, lead - 1))  # cycles of warning before the failing one
            if first_flag_ts is not None:
                leads_seconds.append(max(0.0, float(r["ts"]) - float(first_flag_ts)))
        quiet = 0
    kpis["defect_episodes"] = len(leads_cycles)
    kpis["episodes_with_warning"] = sum(1 for c in leads_cycles if c > 0)
    kpis["median_warning_cycles"] = float(np.median(leads_cycles)) if leads_cycles else None
    kpis["median_warning_seconds"] = round(float(np.median(leads_seconds)), 3) if leads_seconds else None
    return kpis
