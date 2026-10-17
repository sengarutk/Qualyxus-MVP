from __future__ import annotations

import random
from pathlib import Path
import shutil

import pandas as pd
import cv2
from tqdm import tqdm
from dotenv import load_dotenv

from src.config.config import Config
from src.dataset.synthetic_generator import inject_synthetic_data
from src.dataset.yolo_convert import rle_decode, mask_to_polygons, polygon_to_yolo_seg_line


def write_dataset_yaml(yaml_path: Path, dataset_root: Path):
    yaml_text = f"""# Severstal dataset for YOLOv8 segmentation
path: {dataset_root.resolve().as_posix()}
train: images/train
val: images/val

names:
  0: defect
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text, encoding="utf-8")


def prepare():
    load_dotenv()
    cfg = Config.load()
    cfg.ensure_dirs()

    train_csv = Path(cfg.get("severstal", "train_csv"))
    images_dir = Path(cfg.get("severstal", "train_images_dir"))

    processed_dir = Path(cfg.get("paths", "processed_dir"))
    yaml_path = Path(cfg.get("dataset", "yolo", "yaml_path"))

    val_split = float(cfg.get("dataset", "yolo", "val_split"))
    seed = int(cfg.get("dataset", "yolo", "seed"))

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv at: {train_csv}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing train_images dir at: {images_dir}")

    # Ensure dirs
    for p in [
        processed_dir / "images" / "train",
        processed_dir / "images" / "val",
        processed_dir / "labels" / "train",
        processed_dir / "labels" / "val",
    ]:
        p.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv)
    df = df.dropna(subset=["EncodedPixels"]).copy()

    # Robust handling for Severstal CSV formats
    if "ImageId_ClassId" in df.columns:
        # Original Kaggle format: ImageId_ClassId = <image>_<class>
        df["ImageId"] = df["ImageId_ClassId"].str.split("_").str[0]
    elif "ImageId" in df.columns:
        # New / cleaned format (your case) → ImageId already present
        pass
    else:
        raise RuntimeError(
            f"Unknown Severstal CSV format. Columns found: {df.columns.tolist()}"
        )

    grouped = df.groupby("ImageId")["EncodedPixels"].apply(list).reset_index()

    image_ids = grouped["ImageId"].tolist()
    random.Random(seed).shuffle(image_ids)

    n_val = int(len(image_ids) * val_split)
    val_ids = set(image_ids[:n_val])

    write_dataset_yaml(yaml_path, processed_dir)

    print(f"Total defect images: {len(image_ids)} | Train: {len(image_ids)-len(val_ids)} | Val: {len(val_ids)}")
    print(f"Writing dataset yaml: {yaml_path}")

    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Preparing Severstal YOLOv8-seg dataset"):
        image_id = row["ImageId"]
        rles = row["EncodedPixels"]

        img_path = images_dir / image_id
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        union_mask = None
        for rle in rles:
            mask = rle_decode(rle, (h, w))
            union_mask = mask if union_mask is None else (union_mask | mask)

        polys = mask_to_polygons(union_mask, min_area=80)
        if not polys:
            continue

        # train/val split
        split = "val" if image_id in val_ids else "train"
        out_img = processed_dir / "images" / split / image_id
        out_lbl = processed_dir / "labels" / split / (Path(image_id).stem + ".txt")

        shutil.copyfile(img_path, out_img)

        lines = []
        for poly in polys:
            seg_line = polygon_to_yolo_seg_line(poly, w, h)
            lines.append(f"0 {seg_line}")  # class 0

        out_lbl.write_text("\n".join(lines), encoding="utf-8")

    print("[QUALYXUS] ✅ Dataset preparation complete.")
    print("[QUALYXUS] Processed dataset saved under:", processed_dir)


def prepare_with_synthetic(synthetic_path: str | None = None, ratio: float = 0.8):
    prepare()
    if synthetic_path:
        cfg = Config.load()
        processed_dir = Path(cfg.get("paths", "processed_dir"))
        summary = inject_synthetic_data(processed_dir, synthetic_path, ratio=ratio)
        print(f"[QUALYXUS] ✅ Synthetic data injected: {summary}")


if __name__ == "__main__":
    prepare()
