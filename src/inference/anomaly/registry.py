"""Create / load any anomaly detector by name or from a saved file."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.inference.anomaly.base import FORMAT, AnomalyDetector

# Candidate recipes considered by automatic model selection, cheapest first.
CANDIDATES: dict[str, dict[str, Any]] = {
    "patchcore_wrn50": {"method": "patchcore", "backbone": "wide_resnet50_2", "image_size": 256},
    "patchcore_dinov2_s": {"method": "patchcore", "backbone": "dinov2_vits14", "image_size": 448},
    "patchcore_dinov2reg_b": {"method": "patchcore", "backbone": "dinov2reg_vitb14", "image_size": 448},
    "dinomaly_dinov2reg_b": {"method": "dinomaly", "backbone": "dinov2reg_vitb14", "image_size": 392},
    # Score fusion of the two strongest single models on VisA (local texture + structural defects).
    "ensemble_dinomaly_patchcore_s": {"method": "ensemble",
                                      "members": ["dinomaly_dinov2reg_b", "patchcore_dinov2_s"]},
}


def detector_class(kind: str):
    if kind == "patchcore":
        from src.inference.anomaly.patchcore import PatchCoreDetector

        return PatchCoreDetector
    if kind == "dinomaly":
        from src.inference.anomaly.dinomaly import DinomalyDetector

        return DinomalyDetector
    if kind == "ensemble":
        from src.inference.anomaly.ensemble import EnsembleDetector

        return EnsembleDetector
    raise ValueError(f"Unknown anomaly model kind '{kind}'")


def create_detector(method: str, **kwargs) -> AnomalyDetector:
    method = method.lower()
    if method == "patchcore":
        from src.inference.anomaly.patchcore import PatchCoreDetector

        return PatchCoreDetector(**kwargs)
    if method == "dinomaly":
        from src.inference.anomaly.dinomaly import DinomalyDetector

        return DinomalyDetector(**kwargs)
    raise ValueError(f"Unknown anomaly method '{method}' (patchcore | dinomaly)")


def create_candidate(name: str, device: str | None = None, *, register: bool = False,
                     tiles: tuple[int, int] | None = None) -> tuple[AnomalyDetector, dict[str, Any]]:
    spec = dict(CANDIDATES[name])
    method = spec.pop("method")
    if method == "ensemble":
        from src.inference.anomaly.ensemble import EnsembleDetector

        detector = EnsembleDetector([create_candidate(m, device)[0] for m in spec["members"]], device=device)
    else:
        detector = create_detector(method, device=device, **spec)
    if register or tiles:
        detector.configure_preprocessing(register=register, tiles=tiles)
    return detector, {"method": method, **spec, "register": register, "tiles": tiles}


def load_detector(path: str | Path, *, device: str | None = None) -> AnomalyDetector:
    if str(path).lower().endswith(".onnx"):
        from src.inference.anomaly.export import OnnxDetector

        return OnnxDetector(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    fmt = payload.get("format")
    if fmt == "qualyxus-patchcore-v1":
        from src.inference.anomaly.patchcore import PatchCoreDetector

        return PatchCoreDetector.load(path, device=device)
    if fmt != FORMAT:
        raise ValueError(f"{path} is not a Qualyxus anomaly model")
    det = detector_class(payload.get("kind"))._from_state(payload["state"], device)
    det._restore_common(payload)
    return det
