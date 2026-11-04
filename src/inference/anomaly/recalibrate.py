"""Recalibrate a trained anomaly model so its false-reject target holds per part, not per view.

A threshold calibrated on single images bounds the false rejects of one camera. A part seen
by N cameras gets N chances to be falsely rejected, so 1.6% per view becomes about 9% per
part on a six-view station. This splits the part-level budget across the views (union
bound) and recalibrates on good images the model has never seen:

    python -m src.inference.anomaly.recalibrate --model models/anomaly/housing.pt \\
        --images data/raw/housing_holdout --views 6 --out models/anomaly/housing_6view.pt

    python -m src.inference.anomaly.recalibrate --model models/anomaly/synthetic_parts.pt \\
        --synthetic 1500 --views 6 --out models/anomaly/synthetic_parts_6view.pt

A part-level 1% across six views needs at least 599 held-out good images.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

logger = logging.getLogger("AnomalyRecalibrate")

# Far outside the seeds training renders (10,000 + i), so calibration images are unseen.
SYNTHETIC_HOLDOUT_SEED = 200_000


def recalibrate(detector, good_images: Sequence[np.ndarray], *, part_frr: float = 0.01,
                n_views: int = 1) -> dict[str, Any]:
    """Set the detector's threshold so a good part is falsely rejected at most ``part_frr``."""
    if n_views < 1:
        raise ValueError("n_views must be at least 1")
    result = detector.calibrate(detector._holdout_scores(good_images), target_frr=part_frr / n_views)
    detector.calibration.update({"part_frr_target": part_frr, "n_views": n_views})
    return {**result, "part_frr_target": part_frr, "n_views": n_views}


def main(argv=None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, help="Trained anomaly model (.pt)")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", help="Directory of held-out good-part images, unseen in training")
    src.add_argument("--synthetic", type=int, help="Render N unseen simulated good parts")
    parser.add_argument("--views", type=int, required=True, help="Cameras that judge each part")
    parser.add_argument("--part-frr", type=float, default=0.01, help="Target false rejects per part")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    from src.inference.anomaly.registry import load_detector

    detector = load_detector(args.model)
    if args.synthetic:
        from src.process.synthetic_parts import render_part

        images = [render_part(args.width, args.height, None, seed=SYNTHETIC_HOLDOUT_SEED + i)
                  for i in range(args.synthetic)]
    else:
        paths = sorted(p for p in Path(args.images).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"})
        images = [img for img in (cv2.imread(str(p)) for p in paths) if img is not None]
    result = recalibrate(detector, images, part_frr=args.part_frr, n_views=args.views)
    detector.save(args.out)
    logger.info("Saved %s | calibration %s", args.out, json.dumps(result, default=str))
    return result


if __name__ == "__main__":
    main()
