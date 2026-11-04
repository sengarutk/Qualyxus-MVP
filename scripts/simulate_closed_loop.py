#!/usr/bin/env python3
"""Closed-loop validation of the vision <-> process-telemetry flywheel (simulation).

What it does, cycle by cycle, against the injection-molding simulator:

1. The machine produces a part; the process service extracts features and scores its
   defect risk *before* the part reaches the camera.
2. The part is rendered as the inspection camera sees it and judged by the
   good-parts-only PatchCore model (or by ground truth with ``--vision oracle``).
3. The verdict is linked back to its cycle by *time-based* alignment (with transport
   jitter), becoming a training label.
4. After a learning phase the supervised risk model is trained; during the evaluation
   phase high-risk cycles produce setpoint prescriptions which a simulated operator applies
   after a short delay.
5. A twin simulator with the *identical* disturbance history runs without any intervention:
   scrap reduction is measured against that true counterfactual, not against a guess.

    python scripts/simulate_closed_loop.py --learn 2500 --evaluate 2500 --vision patchcore
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import argparse
import collections
import json
import logging
import shutil
import tempfile
import time

import numpy as np

from src.process.models import FAIL, PASS
from src.process.service import ProcessIntelligenceService
from src.process.simulator import InjectionMoldingSimulator
from src.process.synthetic_parts import render_part

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ClosedLoop")


def _rate(n: int, d: int) -> float | None:
    return round(n / d, 5) if d else None


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--learn", type=int, default=2500, help="Cycles observed before prescriptions are acted on")
    parser.add_argument("--evaluate", type=int, default=2500, help="Cycles with the closed loop active")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--vision", choices=["patchcore", "oracle"], default="patchcore")
    parser.add_argument("--transport-delay", type=float, default=4.0)
    parser.add_argument("--jitter", type=float, default=0.3, help="Std-dev of transport time jitter (s)")
    parser.add_argument("--operator-delay", type=int, default=5, help="Cycles before a prescription is applied")
    parser.add_argument("--retrain-every", type=int, default=500)
    parser.add_argument("--out", type=str, default="reports/closed_loop_report.json")
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args(argv)

    workdir = Path(tempfile.mkdtemp(prefix="qualyxus_closed_loop_"))
    rng = np.random.default_rng(args.seed + 1)
    start_ts = 1_700_000_000.0

    detector = None
    if args.vision == "patchcore":
        from src.inference.anomaly.patchcore import PatchCoreDetector

        print("[1/4] Training PatchCore on 120 rendered good parts ...", flush=True)
        detector = PatchCoreDetector(backbone="wide_resnet50_2", image_size=256)
        detector.fit([render_part(640, 480, None, seed=500_000 + i) for i in range(120)])

    cfg = {
        "db_path": str(workdir / "process.db"),
        "models_dir": str(workdir / "models"),
        "model": {"golden_warmup_cycles": 50, "retrain_every_labels": 10**9,
                  "min_train_samples": 300, "min_train_positives": 20},
        "prescriptions": {"enabled": True},
    }
    cams = [{"id": "cam_process", "process_link": {"machine_id": "imm_1", "mode": "time",
                                                    "transport_delay_s": args.transport_delay,
                                                    "tolerance_s": max(1.0, 5 * args.jitter)}}]
    service = ProcessIntelligenceService(cfg, cameras_cfg=cams)
    controlled = InjectionMoldingSimulator(seed=args.seed, start_ts=start_ts)
    baseline = InjectionMoldingSimulator(seed=args.seed, start_ts=start_ts)

    issued: list = []
    service.add_alert_callback(lambda a, c, presc: issued.extend(presc))
    pending: list[tuple[int, object]] = []
    applied: collections.Counter = collections.Counter()

    stats = {phase: collections.Counter() for phase in ("learn", "evaluate")}
    eval_risk: list[tuple[float, int]] = []
    defect_mix = {phase: collections.Counter() for phase in ("controlled", "baseline")}
    retrain_reports = []
    total = args.learn + args.evaluate
    print(f"[2/4] Running {total} cycles ({args.vision} vision) ...", flush=True)
    t0 = time.time()

    for i in range(total):
        phase = "learn" if i < args.learn else "evaluate"
        if i == args.learn:
            retrain_reports.append(service.retrain())
            print(f"      learning phase done; active model: {service.model.kind if service.model else None}", flush=True)
        elif i > args.learn and (i - args.learn) % args.retrain_every == 0:
            retrain_reports.append(service.retrain())

        # Operator applies due prescriptions (only once the loop is active).
        if phase == "evaluate":
            due = [p for when, p in pending if when <= i]
            pending = [(when, p) for when, p in pending if when > i]
            for presc in due:
                controlled.apply_setpoints({presc.setpoint: presc.recommended})
                applied[presc.setpoint] += 1

        cycle, outcome = controlled.next_cycle()
        base_cycle, base_outcome = baseline.next_cycle()
        n_before = len(issued)
        assessment = service.on_cycle(cycle)
        if phase == "evaluate":
            pending.extend((i + args.operator_delay, p) for p in issued[n_before:])
            if assessment is not None:
                eval_risk.append((assessment.risk, int(outcome.is_defective)))
            defect_mix["controlled"][outcome.defect_type or "none"] += 1
            defect_mix["baseline"][base_outcome.defect_type or "none"] += 1

        # Vision inspection at the downstream station.
        if detector is not None:
            result = detector.predict(render_part(640, 480, outcome.defect_type, seed=cycle.sequence))
            verdict, score = (FAIL if result.is_anomalous else PASS), result.normalized_score
        else:
            verdict, score = (FAIL if outcome.is_defective else PASS), None
        seen_ts = cycle.end_ts + args.transport_delay + float(rng.normal(0.0, args.jitter))
        record = service.record_inspection("cam_process", seen_ts, verdict, anomaly_score=score,
                                           defect_type="anomaly" if verdict == FAIL else None)

        s = stats[phase]
        s["parts"] += 1
        s["cycle_time_controlled"] += cycle.duration_s + 1.5
        s["cycle_time_baseline"] += base_cycle.duration_s + 1.5
        s["true_defects"] += int(outcome.is_defective)
        s["baseline_defects"] += int(base_outcome.is_defective)
        s["vision_fail"] += int(verdict == FAIL)
        s["vision_tp"] += int(verdict == FAIL and outcome.is_defective)
        s["vision_fp"] += int(verdict == FAIL and not outcome.is_defective)
        s["vision_fn"] += int(verdict == PASS and outcome.is_defective)
        s["aligned_correctly"] += int(record.cycle_id == cycle.cycle_id)
        if (i + 1) % 1000 == 0:
            print(f"      {i + 1}/{total} cycles ({time.time() - t0:.0f}s)", flush=True)

    print("[3/4] Computing report ...", flush=True)
    ev, ln = stats["evaluate"], stats["learn"]
    risk_auc = None
    if eval_risk and len({y for _, y in eval_risk}) == 2:
        from sklearn.metrics import roc_auc_score

        risk_auc = round(float(roc_auc_score([y for _, y in eval_risk], [r for r, _ in eval_risk])), 4)
    kpis = service.kpis(window_s=10**10)
    both = ln + ev
    report = {
        "config": vars(args),
        "vision": {
            "precision": _rate(both["vision_tp"], both["vision_tp"] + both["vision_fp"]),
            "recall": _rate(both["vision_tp"], both["vision_tp"] + both["vision_fn"]),
            "false_reject_rate": _rate(both["vision_fp"], both["parts"] - both["true_defects"]),
        },
        "alignment_accuracy": _rate(both["aligned_correctly"], both["parts"]),
        "risk_model": {
            "active": service.model.kind if service.model else None,
            "validation": (service.model.metrics.get("validation") if service.model else None),
            "evaluation_phase_auc_vs_ground_truth": risk_auc,
            "median_warning_cycles": kpis.get("median_warning_cycles"),
            "defect_episodes_with_warning": f"{kpis.get('episodes_with_warning')}/{kpis.get('defect_episodes')}",
        },
        "scrap": {
            "learning_phase_rate": _rate(ln["true_defects"], ln["parts"]),
            "evaluation_baseline_rate": _rate(ev["baseline_defects"], ev["parts"]),
            "evaluation_closed_loop_rate": _rate(ev["true_defects"], ev["parts"]),
            "relative_reduction": (
                round(1.0 - ev["true_defects"] / ev["baseline_defects"], 4) if ev["baseline_defects"] else None
            ),
            "defect_mix": {k: dict(v) for k, v in defect_mix.items()},
        },
        "throughput": {
            # Scrap gains bought with slower cycles must be visible: good parts per hour.
            "baseline_cycle_time_s": round(ev["cycle_time_baseline"] / max(ev["parts"], 1), 3),
            "closed_loop_cycle_time_s": round(ev["cycle_time_controlled"] / max(ev["parts"], 1), 3),
            "baseline_good_parts_per_hour": round(
                (ev["parts"] - ev["baseline_defects"]) / (ev["cycle_time_baseline"] / 3600.0), 2) if ev["parts"] else None,
            "closed_loop_good_parts_per_hour": round(
                (ev["parts"] - ev["true_defects"]) / (ev["cycle_time_controlled"] / 3600.0), 2) if ev["parts"] else None,
        },
        "prescriptions": {"issued": len(issued), "applied": dict(applied)},
        "retrains": [{k: v for k, v in r.items() if k != "candidate"} for r in retrain_reports],
        "runtime_s": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("[4/4] Report written to", out)
    print(json.dumps({k: report[k] for k in ("vision", "alignment_accuracy", "risk_model", "scrap", "throughput", "prescriptions")},
                     indent=2, default=str))
    if args.keep_workdir:
        print("Work directory kept at", workdir)
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return report


if __name__ == "__main__":
    main()
