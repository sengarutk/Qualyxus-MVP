from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config.config import Config
from src.inference.postprocess import overlay_mask, bbox_from_mask, severity_from_conf
from src.logging.event_logger import EventLogger
from src.logging.schema import DefectEvent
from src.telemetry.telemetry_provider import TelemetryProvider


def resolve_default_weights(cfg: Config) -> Path:
    """
    Resolve trained model weights.
    Priority:
    1) runs/ (Ultralytics default)
    2) models/checkpoints (manual export)
    """

    # 1. Check Ultralytics runs directory
    runs_path = Path("runs/segment/models/checkpoints/severstal_yolov8_seg/weights/best.pt")
    if runs_path.exists():
        return runs_path.resolve()

    # 2. Fallback to models directory
    models_path = (
        Path(cfg.get("paths", "models_dir"))
        / "severstal_yolov8_seg"
        / "weights"
        / "best.pt"
    )

    if models_path.exists():
        return models_path.resolve()

    raise FileNotFoundError(
        "Could not find trained model weights in runs/ or models/. "
        "Train the model first."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=None)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--source", type=str, default="line_1_cam_1")
    parser.add_argument("--every_n", type=int, default=10)
    args = parser.parse_args()

    cfg = Config.load()
    cfg.ensure_dirs()

    save_dir = Path("data/live/defects") / args.source
    save_dir.mkdir(parents=True, exist_ok=True)

    logger = EventLogger(
        sqlite_path=cfg.get("logging", "sqlite_path"),
        csv_path=cfg.get("logging", "csv_path"),
    )

    # Telemetry
    telemetry = TelemetryProvider(simulation_mode=True)

    conf = float(cfg.get("inference", "conf"))
    iou = float(cfg.get("inference", "iou"))

    weights_path = Path(args.weights) if args.weights else resolve_default_weights(cfg)
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    model = YOLO(str(weights_path))

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video_path}")

    frame_idx = 0
    saved_events = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        if frame_idx % args.every_n != 0:
            continue

        result = model.predict(frame, conf=conf, iou=iou, verbose=False)[0]
        if result.masks is None or result.masks.data is None:
            continue

        masks = result.masks.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.array([])

        for i, mask in enumerate(masks):
            c = float(confs[i]) if i < len(confs) else 0.0
            if c < conf:
                continue

            mask_bin = (mask > 0.5).astype(np.uint8)

            overlay = overlay_mask(frame, mask_bin)
            overlay_path = save_dir / f"frame_{frame_idx}_overlay_{i}.jpg"
            cv2.imwrite(str(overlay_path), overlay)

            crop_path = None
            bbox = bbox_from_mask(mask_bin)
            if bbox is not None:
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = bbox
                x1 = int(np.clip(x1, 0, w - 1))
                y1 = int(np.clip(y1, 0, h - 1))
                x2 = int(np.clip(x2, 0, w - 1))
                y2 = int(np.clip(y2, 0, h - 1))
                if x2 >= x1 and y2 >= y1:
                    crop = frame[y1:y2 + 1, x1:x2 + 1]
                    if crop.size > 0:
                        crop_path = save_dir / f"frame_{frame_idx}_crop_{i}.jpg"
                        cv2.imwrite(str(crop_path), crop)

            # Fetch telemetry
            telemetry_ctx = telemetry.get_telemetry(args.source)

            event = DefectEvent(
                event_id=logger.new_event_id(),
                timestamp=logger.now(),
                source=args.source,
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
                full_image_path=str(Path(args.video_path).resolve()),
                overlay_path=str(overlay_path.resolve()),
                crop_path=str(crop_path.resolve()) if crop_path else None,
            )
            logger.log_event(event)
            saved_events += 1

    cap.release()
    print(f"[QUALYXUS] ✅ Video inference complete. Saved events = {saved_events}")


if __name__ == "__main__":
    main()
