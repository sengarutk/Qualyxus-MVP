from __future__ import annotations

from pathlib import Path

from src.utils.profile_manager import ProfileManager


def resolve_model_path(config_path: str = "config.yaml") -> Path:
    """Resolve active model path using profile cartridge first, then legacy fallbacks."""

    pm = ProfileManager(config_path=config_path)
    candidate = pm.get_model_path()
    if candidate.exists():
        return candidate.resolve()

    # Legacy fallback chain to preserve backward compatibility.
    legacy_candidates = [
        Path("runs/segment/models/checkpoints/severstal_yolov8_seg/weights/best.pt"),
        Path("models/checkpoints/severstal_yolov8_seg/weights/best.pt"),
        Path("yolo26n.pt"),
        Path("yolov8s-seg.pt"),
    ]
    for item in legacy_candidates:
        if item.exists():
            return item.resolve()

    return candidate.resolve()
