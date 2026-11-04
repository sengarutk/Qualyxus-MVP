"""Train + evaluate a supervised defect segmentation model and write its model card.

    # Default: RF-DETR-Seg Small (Apache-2.0) on a YOLO-format recipe dataset
    python -m src.inference.segmentation.train --data data/processed/active_recipe.yaml \\
        --out models/segmentation/active_recipe

    # Validate the pipeline on simulator-rendered labelled parts
    python -m src.inference.segmentation.train --synthetic --arch rfdetr-seg-nano --epochs 10 \\
        --out models/segmentation/synthetic_parts

The operating confidence threshold is chosen on the validation split as the highest one
that still catches >= ``--target-recall`` of defective images (zero-escape operation),
falling back to the F1-optimal threshold.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.inference.segmentation.base import ARCHITECTURES, DEFAULT_ARCH, load_segmenter, write_card

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SegmentationTrainer")


def _dataset_root(data_yaml: Path) -> tuple[Path, dict[str, Any]]:
    """Return a directory containing ``data.yaml`` (as RF-DETR expects) and the parsed YAML."""
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(spec.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    if data_yaml.name not in ("data.yaml", "data.yml") or data_yaml.parent.resolve() != root:
        spec = {**spec, "path": str(root)}
        (root / "data.yaml").write_text(yaml.safe_dump(spec), encoding="utf-8")
    names = spec.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return root, {**spec, "names": list(names or [])}


def _load_split(root: Path, spec: dict[str, Any], split: str) -> list[tuple[np.ndarray, np.ndarray, list[int]]]:
    """[(image, union mask, class ids present)] from YOLO-seg polygon labels."""
    img_dir = root / spec.get(split, f"images/{split}")
    lbl_dir = Path(str(img_dir).replace(f"{Path('images')}", "labels", 1))
    if not lbl_dir.exists():
        lbl_dir = root / "labels" / split
    items = []
    for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        classes: list[int] = []
        lbl = lbl_dir / f"{img_path.stem}.txt"
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 7:
                    continue
                classes.append(int(parts[0]))
                pts = (np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
        items.append((img, mask, classes))
    return items


def evaluate(segmenter, items, *, target_recall: float = 0.99) -> dict[str, Any]:
    """Image-level operating metrics + pixel IoU + defect-type accuracy on a labelled split."""
    preds = [segmenter.predict(img, conf=0.01) for img, _, _ in items]
    labels = np.array([1 if classes else 0 for _, _, classes in items])
    image_scores = np.array([max((p.confidence for p in ps), default=0.0) for ps in preds])

    out: dict[str, Any] = {"n_images": int(len(items)), "n_defective": int(labels.sum())}
    if 0 < labels.sum() < len(labels):
        from sklearn.metrics import roc_auc_score

        out["image_auroc"] = float(roc_auc_score(labels, image_scores))

    best_t, best_f1 = 0.5, -1.0
    chosen = None
    for t in np.round(np.arange(0.95, 0.04, -0.05), 2):
        flagged = image_scores >= t
        recall = float(flagged[labels == 1].mean()) if labels.any() else 0.0
        frr = float(flagged[labels == 0].mean()) if (labels == 0).any() else 0.0
        tp, fp, fn = int((flagged & (labels == 1)).sum()), int((flagged & (labels == 0)).sum()), int((~flagged & (labels == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
        if chosen is None and recall >= target_recall:
            chosen = (float(t), recall, frr)
    threshold = chosen[0] if chosen else best_t
    flagged = image_scores >= threshold
    out.update({
        "conf_threshold": threshold,
        "threshold_rule": f"recall>={target_recall}" if chosen else "f1_max",
        "image_recall": float(flagged[labels == 1].mean()) if labels.any() else None,
        "image_frr": float(flagged[labels == 0].mean()) if (labels == 0).any() else None,
    })

    ious, type_hits, type_total = [], 0, 0
    for (img, mask, classes), ps in zip(items, preds):
        if not classes:
            continue
        kept = [p for p in ps if p.confidence >= threshold]
        pred_mask = np.zeros(mask.shape, bool)
        for p in kept:
            if p.mask is not None and p.mask.shape == mask.shape:
                pred_mask |= p.mask
        inter, union = np.logical_and(pred_mask, mask > 0).sum(), np.logical_or(pred_mask, mask > 0).sum()
        ious.append(inter / union if union else 0.0)
        if kept:
            type_total += 1
            type_hits += int(max(kept, key=lambda p: p.confidence).class_id == classes[0])
    out["defect_mask_iou_mean"] = float(np.mean(ious)) if ious else None
    out["defect_type_accuracy"] = type_hits / type_total if type_total else None
    return out


def main(argv=None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", type=str, help="YOLO-seg dataset YAML (recipe yaml or data.yaml)")
    src.add_argument("--synthetic", action="store_true", help="Render a labelled synthetic-parts dataset")
    parser.add_argument("--arch", default=DEFAULT_ARCH, choices=sorted(ARCHITECTURES))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-recall", type=float, default=0.99)
    parser.add_argument("--out", required=True)
    parser.add_argument("--synthetic-train", type=int, default=400)
    parser.add_argument("--synthetic-val", type=int, default=100)
    args = parser.parse_args(argv)

    out = Path(args.out)
    if args.synthetic:
        from src.process.synthetic_parts import export_segmentation_dataset

        data_yaml = Path(export_segmentation_dataset(out / "dataset", n_train=args.synthetic_train, n_val=args.synthetic_val))
    else:
        data_yaml = Path(args.data)
    root, spec = _dataset_root(data_yaml)
    spec_arch = ARCHITECTURES[args.arch]
    if spec_arch["license"] != "Apache-2.0":
        logger.warning("[Segmentation] %s is %s licensed - check before commercial deployment.", args.arch, spec_arch["license"])

    run_dir = out / "run"
    t0 = time.time()
    if spec_arch["backend"] == "rfdetr":
        from src.inference.segmentation.rfdetr_backend import train_rfdetr

        trained = train_rfdetr(args.arch, root, run_dir, epochs=args.epochs, batch_size=args.batch,
                               grad_accum_steps=args.grad_accum, lr=args.lr)
        weights_src = run_dir / trained["weights"]
    else:
        from src.inference.segmentation.yolo_backend import train_yolo

        trained = train_yolo(args.arch, root / "data.yaml", run_dir, epochs=args.epochs, batch_size=args.batch * args.grad_accum)
        weights_src = run_dir / trained["weights"]
    train_seconds = time.time() - t0

    shutil.copy2(weights_src, out / Path(trained["weights"]).name)
    card = {
        "backend": spec_arch["backend"],
        "arch": args.arch,
        "license": spec_arch["license"],
        "weights": Path(trained["weights"]).name,
        "resolution": trained["resolution"],
        "class_names": spec["names"],
        "conf_threshold": 0.5,
        "min_area_px": 0,
        "dataset": str(root),
        "train_seconds": round(train_seconds, 1),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_card(out, card)

    segmenter = load_segmenter(out)
    val_items = _load_split(root, spec, "val")
    metrics = evaluate(segmenter, val_items, target_recall=args.target_recall)
    card.update({"conf_threshold": metrics["conf_threshold"], "validation": metrics})
    write_card(out, card)
    logger.info("Model card: %s", json.dumps(card, indent=2, default=str))
    return card


if __name__ == "__main__":
    main()
