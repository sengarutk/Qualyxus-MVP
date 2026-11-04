"""Benchmark anomaly detectors and pick the best one for a recipe.

    # Real public benchmark (VisA, CC BY 4.0):
    python -m src.inference.anomaly.benchmark --visa data/benchmarks/visa_raw --categories pcb1,capsules

    # A recipe that only has good images (pseudo-defect proxy) - trains the winner and saves it:
    python -m src.inference.anomaly.benchmark --good-dir data/raw/<recipe>/images \\
        --select --save-best models/anomaly/<recipe>.pt

Selection rule (industrial): among candidates meeting the latency budget, minimise the
false-reject rate at zero escapes (FRR @ 100% defect recall), then the FRR @ 99% recall,
then maximise AU-PRO / image AUROC. Every candidate's numbers are kept in the report.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.inference.anomaly.datasets import (
    AnomalyDataset,
    load_folder,
    load_good_only,
    load_synthetic_parts,
    load_visa,
)
from src.inference.anomaly.metrics import image_metrics, pixel_metrics
from src.inference.anomaly.registry import CANDIDATES, create_candidate

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AnomalyBenchmark")


def evaluate_candidate(name: str, dataset: AnomalyDataset, *, train_kwargs: dict | None = None,
                       device: str | None = None, keep_model: bool = False, register: bool = False,
                       tiles: tuple[int, int] | None = None) -> tuple[dict[str, Any], Any]:
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    detector, spec = create_candidate(name, device=device, register=register, tiles=tiles)
    kwargs = dict(train_kwargs or {})
    if spec["method"] not in ("dinomaly", "ensemble"):
        kwargs = {k: v for k, v in kwargs.items() if k in ("batch_size", "seed")}
    t0 = time.time()
    calibration = detector.fit(dataset.train_good, **kwargs)
    train_s = time.time() - t0

    scores, maps, latencies = [], [], []
    detector.infer(dataset.test_images[0])  # warm-up (CUDA kernels, lazy init)
    for img in dataset.test_images:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()
        score, amap = detector.infer(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t) * 1000.0)
        scores.append(score)
        if dataset.has_masks:
            maps.append(amap)

    result: dict[str, Any] = {"candidate": name, **spec}
    result.update(image_metrics(dataset.test_labels, scores))
    if dataset.has_masks:
        defect_idx = [i for i, y in enumerate(dataset.test_labels) if y]
        good_idx = [i for i, y in enumerate(dataset.test_labels) if not y]
        idx = defect_idx + good_idx
        result.update(pixel_metrics([maps[i] for i in idx], [dataset.test_masks[i] for i in idx]))
    thr = detector.threshold
    labels = np.asarray(dataset.test_labels)
    s = np.asarray(scores)
    result.update({
        "calibrated_threshold": thr,
        "calibrated_frr": float(np.mean(s[labels == 0] > thr)) if (labels == 0).any() else None,
        "calibrated_recall": float(np.mean(s[labels == 1] > thr)) if (labels == 1).any() else None,
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
        "train_seconds": round(train_s, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        "n_train": len(dataset.train_good),
        "proxy_defects": dataset.proxy,
        "calibration": calibration,
    })
    model = detector if keep_model else None
    if not keep_model:
        del detector
        gc.collect()
        torch.cuda.empty_cache()
    return result, model


def _selection_key(result: dict[str, Any]) -> tuple:
    def val(key, default):
        v = result.get(key)
        return default if v is None or (isinstance(v, float) and np.isnan(v)) else v
    return (
        val("frr_at_recall_100", 1.0),
        val("frr_at_recall_99", 1.0),
        -val("pixel_aupro", 0.0),
        -val("image_auroc", 0.0),
        val("latency_ms_mean", 1e9),
    )


def select_best(results: list[dict[str, Any]], latency_budget_ms: float | None = None) -> dict[str, Any] | None:
    eligible = [r for r in results if latency_budget_ms is None or r["latency_ms_mean"] <= latency_budget_ms]
    if not eligible:
        logger.warning("No candidate meets the %.1f ms latency budget; ignoring the budget.", latency_budget_ms or 0)
        eligible = results
    return min(eligible, key=_selection_key) if eligible else None


def _table(rows: list[dict[str, Any]]) -> str:
    cols = ["dataset", "candidate", "image_auroc", "pixel_aupro", "frr_at_recall_100", "frr_at_recall_99",
            "calibrated_frr", "calibrated_recall", "latency_ms_mean", "train_seconds", "peak_vram_gb"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--visa", type=str, help="VisA root (with split_csv/1cls.csv)")
    src.add_argument("--folder", type=str, help="MVTec-style dataset folder")
    src.add_argument("--good-dir", type=str, help="Recipe with only good images (pseudo-defect proxy)")
    src.add_argument("--synthetic", action="store_true", help="Simulator-rendered parts")
    parser.add_argument("--categories", type=str, default="pcb1", help="VisA categories, comma separated")
    parser.add_argument("--candidates", type=str, default=",".join(CANDIDATES))
    parser.add_argument("--max-train", type=int, default=None, help="Cap good training images per dataset")
    parser.add_argument("--dinomaly-iters", type=int, default=3000)
    parser.add_argument("--latency-budget-ms", type=float, default=None)
    parser.add_argument("--select", action="store_true", help="Choose the best candidate per dataset")
    parser.add_argument("--save-best", type=str, default=None, help="Retrain the winner on all data and save it here")
    parser.add_argument("--register", action="store_true", help="Register (align) parts before the model")
    parser.add_argument("--tiles", type=str, default=None, help="Tile grid ROWSxCOLS for small defects, e.g. 2x2")
    parser.add_argument("--confidence", type=float, default=None,
                        help="With --save-best: bound the saved model's FRR with this probability")
    parser.add_argument("--out", type=str, default="reports/anomaly_benchmark.json")
    args = parser.parse_args(argv)
    tiles = tuple(int(v) for v in args.tiles.lower().split("x")) if args.tiles else None

    if args.visa:
        datasets = [lambda c=c: load_visa(args.visa, c, max_train=args.max_train) for c in args.categories.split(",")]
    elif args.folder:
        datasets = [lambda: load_folder(args.folder)]
    elif args.good_dir:
        datasets = [lambda: load_good_only(args.good_dir)]
    else:
        datasets = [load_synthetic_parts]

    names = [c.strip() for c in args.candidates.split(",") if c.strip()]
    unknown = set(names) - set(CANDIDATES)
    if unknown:
        parser.error(f"Unknown candidates {sorted(unknown)}; choose from {sorted(CANDIDATES)}")

    out_path = Path(args.out)
    report: dict[str, Any] = {"results": [], "selected": {}}
    if out_path.exists():  # resume: keep finished results
        try:
            report = json.loads(out_path.read_text())
        except Exception:
            pass
    done = {(r["dataset"], r["candidate"]) for r in report["results"]}

    for make in datasets:
        ds = make()
        logger.info("Dataset %s: %d train good, %d test (%d defective)%s", ds.name, len(ds.train_good),
                    len(ds.test_images), sum(ds.test_labels), " [PROXY defects]" if ds.proxy else "")
        for name in names:
            if (ds.name, name) in done:
                continue
            logger.info("Evaluating %s on %s ...", name, ds.name)
            res, _ = evaluate_candidate(name, ds, train_kwargs={"iterations": args.dinomaly_iters, "log_every": 500},
                                        register=args.register, tiles=tiles)
            res["dataset"] = ds.name
            report["results"].append(res)
            logger.info("  -> AUROC %.4f | AU-PRO %s | FRR@100%% %.4f | %.1f ms",
                        res.get("image_auroc", float("nan")), res.get("pixel_aupro"),
                        res.get("frr_at_recall_100", float("nan")), res["latency_ms_mean"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2, default=str))
        if args.select:
            best = select_best([r for r in report["results"] if r["dataset"] == ds.name], args.latency_budget_ms)
            report["selected"][ds.name] = best["candidate"] if best else None
            if best and args.save_best:
                detector, _ = create_candidate(best["candidate"], register=args.register, tiles=tiles)
                extra = {"iterations": args.dinomaly_iters} if best["method"] in ("dinomaly", "ensemble") else {}
                detector.fit(ds.train_good + [img for img, y in zip(ds.test_images, ds.test_labels) if not y],
                             confidence=args.confidence, **extra)
                detector.save(args.save_best)
                logger.info("Saved selected model %s -> %s", best["candidate"], args.save_best)
        del ds
        gc.collect()

    out_path.write_text(json.dumps(report, indent=2, default=str))
    md = _table(report["results"])
    out_path.with_suffix(".md").write_text(md + "\n")
    print(md)
    if report["selected"]:
        print("Selected:", json.dumps(report["selected"], indent=2))
    return report


if __name__ == "__main__":
    main()
