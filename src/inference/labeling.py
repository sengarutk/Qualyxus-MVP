"""Labelling loop: review queue -> labelling tool -> segmentation dataset + review verdicts.

Export (most informative parts first - active learning)::

    python -m src.inference.labeling export --review-dir data/review --out data/labeling/batch_001 \\
        --format labelstudio --limit 200

  Items are ordered by uncertainty: parts whose anomaly score is closest to the decision
  threshold come first (|log(normalized score)| ascending), interleaved across cameras so
  one camera cannot monopolise a batch. Each task carries the model's box as a
  pre-annotation, so labellers correct instead of drawing from scratch.
  ``--format coco`` writes COCO instances (importable into CVAT as "COCO 1.0").

Import (polygons -> YOLO-seg dataset for ``src.inference.segmentation.train --data``)::

    python -m src.inference.labeling import --labels export.json --format labelstudio \\
        --images data/labeling/batch_001/images --out data/datasets/housing_a_seg \\
        --classes short_shot,sink_mark,flash,burn_mark --update-review data/review

  Images with no polygons become background (good) examples. With ``--update-review``
  the human verdict (fail if any polygon, else pass) and defect type are written back to
  the review items, so ``ingest_reviews`` turns them into process audits.
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

logger = logging.getLogger("QualyxusLabeling")


# ------------------------------------------------------------------- export

def review_items(review_dir: str | Path, *, include_reviewed: bool = False) -> list[dict[str, Any]]:
    items = []
    for meta_path in Path(review_dir).rglob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "review" not in meta:
            continue
        if not include_reviewed and (meta.get("review") or {}).get("verdict"):
            continue
        image = meta_path.with_suffix(".png")
        if image.exists():
            items.append({**meta, "_image": image, "_meta": meta_path})
    return items


def uncertainty(item: dict[str, Any]) -> float:
    """0 = exactly at the threshold (most informative); grows with distance in log-score."""
    s = item.get("normalized_score")
    if s is None or s <= 0:
        return math.inf
    return abs(math.log(float(s)))


def prioritise(items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_camera: dict[str, list] = collections.defaultdict(list)
    for it in sorted(items, key=uncertainty):
        by_camera[it.get("camera_id") or "?"].append(it)
    out, queues = [], [collections.deque(v) for _, v in sorted(by_camera.items())]
    while any(queues):
        for q in queues:
            if q:
                out.append(q.popleft())
    return out


def export_batch(review_dir: str | Path, out_dir: str | Path, *, fmt: str = "labelstudio", limit: int = 200,
                 class_names: list[str] | None = None, include_reviewed: bool = False) -> dict[str, Any]:
    items = prioritise(review_items(review_dir, include_reviewed=include_reviewed))[:limit]
    out = Path(out_dir)
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    tasks, coco_images, coco_anns = [], [], []
    for idx, it in enumerate(items, 1):
        name = f"{it['id']}.png"
        shutil.copy2(it["_image"], img_dir / name)
        h, w = cv2.imread(str(it["_image"])).shape[:2]
        bbox = it.get("bbox")
        label = it.get("predicted_type") or "anomaly"
        if fmt == "labelstudio":
            task = {"data": {"image": f"/data/local-files/?d=images/{name}", "review_id": it["id"],
                             "camera_id": it.get("camera_id"), "normalized_score": it.get("normalized_score"),
                             "model_verdict": it.get("model_verdict")}}
            if bbox:
                x1, y1, x2, y2 = bbox
                task["predictions"] = [{"model_version": "qualyxus", "score": it.get("normalized_score"), "result": [{
                    "from_name": "label", "to_name": "image", "type": "rectanglelabels",
                    "original_width": w, "original_height": h,
                    "value": {"x": 100 * x1 / w, "y": 100 * y1 / h, "width": 100 * (x2 - x1) / w,
                              "height": 100 * (y2 - y1) / h, "rectanglelabels": [label]}}]}]
            tasks.append(task)
        else:
            coco_images.append({"id": idx, "file_name": name, "width": w, "height": h})
            if bbox:
                x1, y1, x2, y2 = bbox
                names = class_names or [label]
                cid = names.index(label) + 1 if label in names else 1
                coco_anns.append({"id": len(coco_anns) + 1, "image_id": idx, "category_id": cid,
                                  "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0,
                                  "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]]})
    if fmt == "labelstudio":
        (out / "tasks.json").write_text(json.dumps(tasks, indent=2), encoding="utf-8")
        names = class_names or sorted({it.get("predicted_type") or "anomaly" for it in items} | {"anomaly"})
        labels = "\n".join(f'    <Label value="{n}"/>' for n in names)
        (out / "labeling_config.xml").write_text(
            "<View>\n  <Image name=\"image\" value=\"$image\" zoom=\"true\"/>\n"
            f"  <PolygonLabels name=\"label\" toName=\"image\">\n{labels}\n  </PolygonLabels>\n</View>\n",
            encoding="utf-8")
    else:
        names = class_names or sorted({it.get("predicted_type") or "anomaly" for it in items})
        coco = {"images": coco_images, "annotations": coco_anns,
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
        (out / "annotations.json").write_text(json.dumps(coco, indent=2), encoding="utf-8")
    summary = {"items": len(items), "format": fmt, "out": str(out),
               "mean_uncertainty": float(np.mean([uncertainty(i) for i in items])) if items else None}
    (out / "batch.json").write_text(json.dumps({**summary, "review_ids": [i["id"] for i in items]}, indent=2),
                                    encoding="utf-8")
    return summary


# ------------------------------------------------------------------- import

def _polygons_from_labelstudio(task: dict[str, Any]) -> tuple[str, list[tuple[str, list[tuple[float, float]]]]]:
    """(image file name, [(class, [(x, y) normalised 0-1])]) from one Label Studio export task."""
    data = task.get("data", {})
    # Our own exports carry the review id; uploads into Label Studio get a "<hash>-" name prefix.
    image = f"{data['review_id']}.png" if data.get("review_id") else str(data.get("image", "")).split("/")[-1].split("=")[-1]
    polys = []
    annotations = task.get("annotations") or []
    results = annotations[-1].get("result", []) if annotations else []
    for r in results:
        v = r.get("value", {})
        if r.get("type") == "polygonlabels" and v.get("points"):
            polys.append((v["polygonlabels"][0], [(x / 100.0, y / 100.0) for x, y in v["points"]]))
        elif r.get("type") == "rectanglelabels":
            x, y, w, h = v["x"] / 100, v["y"] / 100, v["width"] / 100, v["height"] / 100
            polys.append((v["rectanglelabels"][0], [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]))
    return image, polys


def _polygons_from_coco(coco: dict[str, Any]) -> dict[str, list[tuple[str, list[tuple[float, float]]]]]:
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    out: dict[str, list] = {im["file_name"]: [] for im in coco["images"]}
    for ann in coco["annotations"]:
        im = images[ann["image_id"]]
        w, h = float(im["width"]), float(im["height"])
        seg = ann.get("segmentation")
        if isinstance(seg, list) and seg:
            for poly in seg:
                pts = [(poly[i] / w, poly[i + 1] / h) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    out[im["file_name"]].append((cats[ann["category_id"]], pts))
        elif ann.get("bbox"):
            x, y, bw, bh = ann["bbox"]
            out[im["file_name"]].append((cats[ann["category_id"]], [(x / w, y / h), ((x + bw) / w, y / h),
                                                                    ((x + bw) / w, (y + bh) / h), (x / w, (y + bh) / h)]))
    return out


def import_labels(labels_path: str | Path, images_dir: str | Path, out_dir: str | Path, *, fmt: str,
                  class_names: list[str], val_fraction: float = 0.2, seed: int = 0,
                  update_review: str | Path | None = None) -> dict[str, Any]:
    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    if fmt == "labelstudio":
        per_image = dict(_polygons_from_labelstudio(t) for t in data)
    else:
        per_image = _polygons_from_coco(data)
    out = Path(out_dir)
    names = list(class_names)
    files = sorted(per_image)
    random.Random(seed).shuffle(files)
    n_val = max(1, int(round(len(files) * val_fraction))) if len(files) > 1 else 0
    stats = collections.Counter()
    unknown = set()
    for i, fname in enumerate(files):
        split = "val" if i < n_val else "train"
        src = Path(images_dir) / fname
        if not src.exists() and "-" in fname:
            src = Path(images_dir) / fname.split("-", 1)[1]
        if not src.exists():
            stats["missing_images"] += 1
            continue
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / "images" / split / fname)
        lines = []
        for cls, pts in per_image[fname]:
            if cls not in names:
                unknown.add(cls)
                continue
            lines.append(" ".join([str(names.index(cls))] + [f"{min(max(v, 0.0), 1.0):.6f}" for p in pts for v in p]))
            stats[f"class:{cls}"] += 1
        (out / "labels" / split / (Path(fname).stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        stats[f"{split}_images"] += 1
        stats["defective_images" if lines else "good_images"] += 1
        if update_review:
            _write_review_verdict(update_review, Path(fname).stem, per_image[fname], names)
            stats["review_updated"] += 1
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)), encoding="utf-8")
    if unknown:
        logger.warning("Ignored polygons with classes not in --classes: %s", sorted(unknown))
    return {"data_yaml": str(out / "data.yaml"), **dict(stats), "unknown_classes": sorted(unknown)}


def _write_review_verdict(review_dir: str | Path, review_id: str, polys, names: list[str]) -> None:
    for meta_path in Path(review_dir).rglob(f"{review_id}.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        classes = [c for c, _ in polys if c in names]
        meta["review"] = {"verdict": "fail" if classes else "pass",
                          "defect_type": collections.Counter(classes).most_common(1)[0][0] if classes else None,
                          "reviewer": "labeling-import"}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main(argv=None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("--review-dir", default="data/review")
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["labelstudio", "coco"], default="labelstudio")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--classes", default=None)
    p.add_argument("--include-reviewed", action="store_true")
    p = sub.add_parser("import")
    p.add_argument("--labels", required=True)
    p.add_argument("--format", choices=["labelstudio", "coco"], required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--classes", required=True)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--update-review", default=None)
    args = parser.parse_args(argv)
    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    if args.cmd == "export":
        result = export_batch(args.review_dir, args.out, fmt=args.format, limit=args.limit, class_names=classes,
                              include_reviewed=args.include_reviewed)
    else:
        result = import_labels(args.labels, args.images, args.out, fmt=args.format, class_names=classes,
                               val_fraction=args.val_fraction, update_review=args.update_review)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
