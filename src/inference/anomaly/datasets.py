"""Evaluation datasets for anomaly detectors.

* VisA (Amazon, CC BY 4.0) - 12 real industrial categories with pixel masks.
* MVTec-style folders (``train/good``, ``test/<type>``, ``ground_truth/<type>/*_mask.png``) -
  the layout used by MVTec AD/AD 2 and by Qualyxus recipe exports.
* Recipe catalogs with only good images - validated with pseudo-defects (see
  :mod:`src.inference.anomaly.pseudo_defects`), clearly marked as a proxy.

Images are decoded once and kept downscaled (longest side ``max_side``) so a full category
fits in memory.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class AnomalyDataset:
    name: str
    train_good: list[np.ndarray]
    test_images: list[np.ndarray]
    test_labels: list[int]
    test_masks: list[np.ndarray] = field(default_factory=list)  # empty if no pixel ground truth
    proxy: bool = False  # True when defects are synthetic pseudo-anomalies

    @property
    def has_masks(self) -> bool:
        return len(self.test_masks) == len(self.test_images) and len(self.test_masks) > 0


def _read(path: Path, max_side: int, gray: bool = False) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        interp = cv2.INTER_NEAREST if gray else cv2.INTER_AREA
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=interp)
    return img


def load_visa(root: str | Path, category: str, *, max_side: int = 512, max_train: int | None = None,
              seed: int = 0) -> AnomalyDataset:
    root = Path(root)
    rows = list(csv.DictReader(open(root / "split_csv" / "1cls.csv", newline="")))
    rows = [r for r in rows if r["object"] == category]
    if not rows:
        raise ValueError(f"Unknown VisA category '{category}'")
    train = [r for r in rows if r["split"] == "train"]
    if max_train and len(train) > max_train:
        idx = np.random.default_rng(seed).choice(len(train), max_train, replace=False)
        train = [train[i] for i in sorted(idx)]
    test = [r for r in rows if r["split"] == "test"]
    train_imgs = [_read(root / r["image"], max_side) for r in train]
    test_imgs, labels, masks = [], [], []
    for r in test:
        img = _read(root / r["image"], max_side)
        test_imgs.append(img)
        anomalous = r["label"] == "anomaly"
        labels.append(int(anomalous))
        if anomalous and r.get("mask"):
            m = _read(root / r["mask"], max_side, gray=True)
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            masks.append((m > 0).astype(np.uint8))
        else:
            masks.append(np.zeros(img.shape[:2], np.uint8))
    return AnomalyDataset(f"visa/{category}", train_imgs, test_imgs, labels, masks)


def load_folder(root: str | Path, *, max_side: int = 512, name: str | None = None) -> AnomalyDataset:
    """MVTec-style layout: train/good, test/<good|type>, ground_truth/<type>/<stem>_mask.png."""
    root = Path(root)
    files = lambda d: sorted(p for p in d.iterdir() if p.suffix.lower() in _EXT) if d.is_dir() else []  # noqa: E731
    train = [_read(p, max_side) for p in files(root / "train" / "good")]
    test_imgs, labels, masks = [], [], []
    for type_dir in sorted(p for p in (root / "test").iterdir() if p.is_dir()):
        for p in files(type_dir):
            img = _read(p, max_side)
            test_imgs.append(img)
            good = type_dir.name == "good"
            labels.append(0 if good else 1)
            mask_path = root / "ground_truth" / type_dir.name / f"{p.stem}_mask.png"
            if not good and mask_path.exists():
                m = _read(mask_path, max_side, gray=True)
                masks.append((cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8))
            else:
                masks.append(np.zeros(img.shape[:2], np.uint8))
    return AnomalyDataset(name or root.name, train, test_imgs, labels, masks)


def load_good_only(directory: str | Path, *, max_side: int = 512, holdout: float = 0.3, n_pseudo: int | None = None,
                   seed: int = 0, name: str | None = None) -> AnomalyDataset:
    """A recipe with only good images: hold some out and add pseudo-defects (proxy metrics)."""
    from src.inference.anomaly.pseudo_defects import make_pseudo_defect

    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in _EXT)
    images = [_read(p, max_side) for p in paths]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(images))
    n_hold = max(5, int(len(images) * holdout))
    hold = [images[i] for i in order[:n_hold]]
    train = [images[i] for i in order[n_hold:]]
    n_pseudo = n_pseudo or len(hold)
    test_imgs, labels, masks = list(hold), [0] * len(hold), [np.zeros(h.shape[:2], np.uint8) for h in hold]
    for k in range(n_pseudo):
        img, mask = make_pseudo_defect(hold[k % len(hold)], rng, source_pool=train)
        test_imgs.append(img)
        labels.append(1)
        masks.append(mask)
    return AnomalyDataset(name or Path(directory).name, train, test_imgs, labels, masks, proxy=True)


def load_synthetic_parts(n_train: int = 120, n_test_good: int = 40, n_test_each: int = 10,
                         width: int = 640, height: int = 480) -> AnomalyDataset:
    from src.process.synthetic_parts import render_part

    train = [render_part(width, height, None, seed=100_000 + i) for i in range(n_train)]
    test, labels = [], []
    for i in range(n_test_good):
        test.append(render_part(width, height, None, seed=200_000 + i))
        labels.append(0)
    for j, d in enumerate(("short_shot", "flash", "sink_mark", "burn_mark")):
        for i in range(n_test_each):
            test.append(render_part(width, height, d, seed=300_000 + 100 * j + i))
            labels.append(1)
    return AnomalyDataset("synthetic_parts", train, test, labels)
