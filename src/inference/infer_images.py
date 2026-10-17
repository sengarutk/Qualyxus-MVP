from __future__ import annotations

import argparse
import re
import hashlib
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import Config
from src.inference.postprocess import overlay_mask, bbox_from_mask, severity_from_conf
from src.logging.event_logger import EventLogger
from src.logging.schema import DefectEvent
from src.telemetry.telemetry_provider import TelemetryProvider


def extract_camera_id(filename: str) -> str:
    """Dynamically extracts camera identifier prefix from image filename."""
    match = re.search(r"^(cam_[a-zA-Z0-9_]+?)(?:_frame|_defect|_mask|\.|$)", filename, re.IGNORECASE)
    if not match:
        match = re.search(r"^(cam_\d+|cam_[a-zA-Z0-9]+)", filename, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "general"


def compute_image_hash(img_array: np.ndarray) -> str:
    """Computes MD5 hash over image byte array for duplicate detection."""
    return hashlib.md5(img_array.tobytes()).hexdigest()


def resolve_default_weights(cfg: Config) -> Path:
    """
    Resolve trained model weights.
    Priority:
    1) <paths.models_dir>/<dataset.name>/weights/best.pt  (output of src.training.train_yolo_seg)
    2) inference.model_path from the active config
    3) legacy Severstal locations under runs/ and models/
    """
    models_dir = Path(cfg.get("paths", "models_dir", default="models/checkpoints") or "models/checkpoints")
    dataset_name = cfg.get("dataset", "name", default="active_product_recipe") or "active_product_recipe"
    candidates = [models_dir / dataset_name / "weights" / "best.pt"]

    configured = cfg.get("inference", "model_path")
    if configured:
        candidates.append(Path(configured))

    candidates += [
        Path("runs/segment/models/checkpoints/severstal_yolov8_seg/weights/best.pt"),
        models_dir / "severstal_yolov8_seg" / "weights" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find trained model weights. Looked in: "
        + ", ".join(str(c) for c in candidates)
        + ". Train the model first or pass --weights."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default="data/live/raw")
    parser.add_argument("--save_dir", type=str, default="data/live/defects")
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()

    cfg = Config.load()
    cfg.ensure_dirs()

    base_save = Path(args.save_dir) if args.save_dir else Path("data/live/defects")
    base_save.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        import logging
        logging.warning(f"[infer_images] Input directory does not exist: {input_dir}. Exiting gracefully.")
        return

    images = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images.extend(input_dir.glob(ext))

    if not images:
        import logging
        logging.warning(f"[infer_images] No images found in: {input_dir}. Exiting gracefully.")
        return

    logger = EventLogger(
        sqlite_path=cfg.get("logging", "sqlite_path", default="data/events/events.db"),
        csv_path=cfg.get("logging", "csv_path", default="data/events/events.csv"),
    )

    # Telemetry
    telemetry = TelemetryProvider(simulation_mode=True)

    conf = float(cfg.get("inference", "conf"))
    iou = float(cfg.get("inference", "iou"))

    weights_path = Path(args.weights) if args.weights else resolve_default_weights(cfg)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    model = YOLO(str(weights_path))
    written_overlay_hashes: dict[str, Path] = {}

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Dynamically extract camera channel ID or fall back to general/CLI source
        cam_id = args.source or extract_camera_id(img_path.name)
        cam_save_dir = base_save / cam_id
        cam_save_dir.mkdir(parents=True, exist_ok=True)

        result = model.predict(img, conf=conf, iou=iou, verbose=False)[0]
        if result.masks is None or result.masks.data is None:
            continue

        masks = result.masks.data.cpu().numpy()  # (N,H,W)
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])

        for i, mask in enumerate(masks):
            c = float(confs[i]) if i < len(confs) else 0.0
            if c < conf:
                continue

            mask_bin = (mask > 0.5).astype(np.uint8)
            overlay = overlay_mask(img, mask_bin)

            # Deduplicate overlay writing based on content hash and reuse path
            overlay_hash = compute_image_hash(overlay)
            if overlay_hash in written_overlay_hashes and written_overlay_hashes[overlay_hash].exists():
                overlay_path = written_overlay_hashes[overlay_hash]
            else:
                overlay_path = cam_save_dir / f"{img_path.stem}_overlay_{i}.jpg"
                cv2.imwrite(str(overlay_path), overlay)
                written_overlay_hashes[overlay_hash] = overlay_path

            crop_path = None
            bbox = bbox_from_mask(mask_bin)
            if bbox is not None:
                h, w = img.shape[:2]
                x1, y1, x2, y2 = bbox
                x1 = int(np.clip(x1, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y2 = int(np.clip(y2, 0, h - 1))
                if x2 >= x1 and y2 >= y1:
                    crop = img[y1:y2 + 1, x1:x2 + 1]
                    if crop.size > 0:
                        crop_path = cam_save_dir / f"{img_path.stem}_crop_{i}.jpg"
                        # Always (re)write: raw frame names repeat across capture sessions, so a
                        # stale crop from an earlier session must never be attached to this event.
                        cv2.imwrite(str(crop_path), crop)

            # Fetch telemetry for extracted camera channel ID
            telemetry_ctx = telemetry.get_telemetry(cam_id)

            event = DefectEvent(
                event_id=logger.new_event_id(),
                timestamp=logger.now(),
                source=cam_id,
                defect_type="defect",
                confidence=c,
                severity=severity_from_conf(c),
                batch_id=None,
                operator_id=None,
                machine_id=telemetry_ctx.machine_id,
                shift_id=telemetry_ctx.shift_id,
                line_id=telemetry_ctx.line_id,
                station_id=telemetry_ctx.station_id,
                temperature=telemetry_ctx.temperature,
                vibration=telemetry_ctx.vibration,
                speed=telemetry_ctx.speed,
                lot_id=telemetry_ctx.lot_id,
                full_image_path=str(img_path.resolve()),
                overlay_path=str(overlay_path.resolve()),
                crop_path=str(crop_path.resolve()) if crop_path else None,
            )
            logger.log_event(event)

    print("[QUALYXUS] ✅ Inference complete. Events stored in CSV + SQLite.")


if __name__ == "__main__":
    main()
