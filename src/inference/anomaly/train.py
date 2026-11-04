"""Train a good-parts-only anomaly model.

    # Recommended: let the benchmark choose (see src.inference.anomaly.benchmark --select --save-best)

    # Explicit choice:
    python -m src.inference.anomaly.train --images data/raw/<recipe>/images \\
        --method dinomaly --backbone dinov2reg_vitb14 --out models/anomaly/<recipe>.pt
    python -m src.inference.anomaly.train --images ... --method patchcore --backbone dinov2_vits14 --out ...

    # Simulator-rendered good parts (synthetic process scene):
    python -m src.inference.anomaly.train --synthetic 120 --out models/anomaly/synthetic_parts.pt

The decision threshold is calibrated on held-out good images with split-conformal
calibration: the false-reject rate on good parts is bounded by ``--target-frr`` when at
least ``1/target_frr - 1`` held-out images are available. That bound holds on average over
calibrations; ``--confidence 0.9`` instead bounds *this* model's FRR with 90% probability
(needs ``log(0.1)/log(1 - target_frr)`` held-out images, e.g. 230 for 1%).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2

from src.inference.anomaly.backbones import CNN_BACKBONES, VIT_CHECKPOINTS
from src.inference.anomaly.registry import CANDIDATES, create_candidate, create_detector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AnomalyTrainer")

_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def load_images(directory: str | Path, limit: int | None = None):
    paths = sorted(p for p in Path(directory).rglob("*") if p.suffix.lower() in _EXTENSIONS)
    images = []
    for path in paths[: limit or None]:
        img = cv2.imread(str(path))
        if img is not None:
            images.append(img)
    return images


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=str, help="Directory of good-part images (searched recursively)")
    src.add_argument("--synthetic", type=int, help="Render N simulated good parts instead")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--method", choices=["patchcore", "dinomaly"], default="patchcore")
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), default=None,
                        help="Use a benchmark recipe (incl. ensembles) instead of --method/--backbone")
    parser.add_argument("--register", action="store_true", help="Register (align) parts before the model")
    parser.add_argument("--tiles", type=str, default=None, help="Tile grid ROWSxCOLS for small defects, e.g. 2x2")
    parser.add_argument("--backbone", default="wide_resnet50_2", choices=sorted(set(CNN_BACKBONES) | set(VIT_CHECKPOINTS)))
    parser.add_argument("--image-size", type=int, default=None, help="Default: 256 (CNN), 448 (ViT PatchCore), 392 (Dinomaly)")
    parser.add_argument("--coreset", type=float, default=0.1, help="PatchCore memory-bank coreset fraction")
    parser.add_argument("--iterations", type=int, default=3000, help="Dinomaly training iterations")
    parser.add_argument("--target-frr", type=float, default=0.01, help="Target false-reject rate on good parts")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Guarantee this deployment's FRR <= target with this probability (e.g. 0.9; needs more holdout)")
    parser.add_argument("--holdout", type=float, default=0.2, help="Fraction of good images held out for calibration")
    parser.add_argument("--width", type=int, default=640, help="Synthetic render width")
    parser.add_argument("--height", type=int, default=480, help="Synthetic render height")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if not args.candidate and args.method == "dinomaly" and args.backbone in CNN_BACKBONES:
        parser.error("Dinomaly needs a DINO ViT backbone (e.g. dinov2reg_vitb14)")

    if args.synthetic:
        from src.process.synthetic_parts import render_part

        images = [render_part(args.width, args.height, None, seed=10_000 + i) for i in range(args.synthetic)]
    else:
        images = load_images(args.images, args.limit)
    logger.info("Training %s/%s on %d good images", args.method, args.backbone, len(images))

    tiles = tuple(int(v) for v in args.tiles.lower().split("x")) if args.tiles else None
    kwargs = {"backbone": args.backbone}
    if args.candidate:
        detector, spec = create_candidate(args.candidate, register=args.register, tiles=tiles)
        train_kwargs = {"iterations": args.iterations} if spec["method"] in ("dinomaly", "ensemble") else {}
    elif args.method == "patchcore":
        kwargs.update(image_size=args.image_size or (256 if args.backbone in CNN_BACKBONES else 448),
                      coreset_fraction=args.coreset)
        train_kwargs = {}
    else:
        kwargs.update(image_size=args.image_size or 392)
        train_kwargs = {"iterations": args.iterations}
    if not args.candidate:
        detector = create_detector(args.method, **kwargs)
        detector.configure_preprocessing(register=args.register, tiles=tiles)
    calibration = detector.fit(images, holdout_fraction=args.holdout, target_frr=args.target_frr,
                               confidence=args.confidence, **train_kwargs)
    detector.save(args.out)
    logger.info("Saved %s | calibration %s", args.out, json.dumps(calibration, default=str))
    return calibration


if __name__ == "__main__":
    main()
