"""ONNX export + ONNX Runtime serving for anomaly models (Dinomaly and PatchCore).

    python -m src.inference.anomaly.export --model models/anomaly/<recipe>.pt --out models/anomaly/<recipe>.onnx \\
        [--check-images data/raw/<recipe>/images]

The network part of each detector is exported as one graph and verified numerically
against PyTorch before the file is accepted:

* Dinomaly - frozen DINO encoder, bottleneck, decoder and the per-pixel cosine anomaly map;
* PatchCore - backbone features and the nearest-neighbour search against the memory bank
  (the bank is a constant inside the graph), giving per-patch scores.

The cheap, shape-dependent tail (blur, resize to the frame, top-k image score) and any part
registration / tiling run in NumPy, identically to the PyTorch detector. The threshold,
calibration and preprocessing state travel in a JSON sidecar. With ``--check-images`` the
exported model's image scores are compared with the PyTorch model's on real images (the
PyTorch model runs the backbone in bf16/fp16 on GPU, ONNX in fp32), and the export is
rejected if they disagree by more than ``--score-rtol``.

Serving uses ONNX Runtime with the best available execution provider
(TensorRT > CUDA > CPU).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import types
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference.anomaly.backbones import to_tensor_batch
from src.inference.anomaly.base import AnomalyDetector
from src.inference.anomaly.preprocess import Preprocessor

logger = logging.getLogger("QualyxusAnomalyExport")


def _freeze_position_embeddings(encoder: nn.Module, size: int, patch: int) -> None:
    """Bake the positional embeddings for the fixed export size into the graph.

    DINOv2 resizes its position embeddings at run time with antialiased bicubic
    interpolation, which ONNX cannot express; an exported model has a fixed input size,
    so the resized embeddings are computed once here and served as a constant.
    """
    embeddings = encoder.embeddings
    n_patches = (size // patch) ** 2
    dim = embeddings.position_embeddings.shape[-1]
    with torch.no_grad():
        probe = torch.zeros(1, 1 + n_patches, dim, dtype=embeddings.position_embeddings.dtype)
        frozen = embeddings.interpolate_pos_encoding(probe, size, size).detach().clone()
    embeddings.register_buffer("_frozen_pos", frozen, persistent=False)
    embeddings.interpolate_pos_encoding = types.MethodType(lambda self_, emb, h, w: self_._frozen_pos, embeddings)


class _DinomalyGraph(nn.Module):
    def __init__(self, det):
        super().__init__()
        det._ensure()
        # Copies: freezing position embeddings / moving to CPU must not touch the live detector.
        self.encoder = copy.deepcopy(det._backbone.model)
        self.head = copy.deepcopy(det.head)
        self.layers = list(det.target_layers)
        self.groups = [list(g) for g in det.groups]
        self.num_prefix = det._backbone.num_prefix
        self.patch = det._backbone.patch_size
        _freeze_position_embeddings(self.encoder, det.image_size, self.patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
        layers = [hidden[i + 1][:, self.num_prefix:, :] for i in self.layers]
        fused = torch.stack(layers, dim=1).mean(dim=1)
        decoded = self.head(fused)
        h = x.shape[-2] // self.patch
        w = x.shape[-1] // self.patch
        maps = []
        for g in self.groups:
            en = torch.stack([layers[i] for i in g], dim=1).mean(dim=1)
            de = torch.stack([decoded[i] for i in g], dim=1).mean(dim=1)
            en = en.permute(0, 2, 1).reshape(en.shape[0], en.shape[2], h, w)
            de = de.permute(0, 2, 1).reshape(de.shape[0], de.shape[2], h, w)
            m = 1.0 - F.cosine_similarity(en, de, dim=1).unsqueeze(1)
            maps.append(F.interpolate(m, size=(256, 256), mode="bilinear", align_corners=True))
        return torch.cat(maps, dim=1).mean(dim=1)


class _PatchCoreGraph(nn.Module):
    """Backbone features + exact nearest-neighbour distance to the memory bank -> (B, h*w)."""

    def __init__(self, det):
        super().__init__()
        backbone = det._ensure_backbone()
        self.vit = det.vit
        self.bank = nn.Parameter(det.memory_bank.detach().float().cpu(), requires_grad=False)
        if self.vit:
            self.encoder = copy.deepcopy(backbone.model)
            self.num_prefix = backbone.num_prefix
            self.layer = int(det.vit_layer)
            _freeze_position_embeddings(self.encoder, det.image_size, backbone.patch_size)
        else:
            self.cnn = copy.deepcopy(backbone.model)
            self.embed_dim = int(det.embed_dim)
            with torch.no_grad():  # adaptive channel pooling is linear: export it as an exact fixed matrix
                _, grid = backbone(torch.zeros(1, 3, det.image_size, det.image_size, device=det.device))
                c = int(backbone(torch.zeros(1, 3, 64, 64, device=det.device))[0].shape[-1])
                pool = F.adaptive_avg_pool1d(torch.eye(c).unsqueeze(1), self.embed_dim).squeeze(1)
            self.pool = nn.Parameter(pool, requires_grad=False)
            self.bank_sq = nn.Parameter((self.bank ** 2).sum(dim=1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.vit:
            hidden = self.encoder(pixel_values=x, output_hidden_states=True).hidden_states
            tokens = hidden[self.layer + 1][:, self.num_prefix:, :]
            feats = F.normalize(tokens, dim=-1)                              # B, N, C
            return 1.0 - torch.matmul(feats, self.bank.t()).max(dim=-1).values
        m = self.cnn
        y = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f2 = m.layer2(m.layer1(y))
        f3 = m.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([f2, f3], dim=1)
        b, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(b, h * w, c)
        feats = torch.matmul(feats, self.pool)
        sq = (feats ** 2).sum(dim=-1, keepdim=True)
        d2 = (sq + self.bank_sq.view(1, 1, -1) - 2.0 * torch.matmul(feats, self.bank.t())).clamp_min(0.0)
        return d2.min(dim=-1).values.sqrt()


def _graph_and_meta(detector) -> tuple[nn.Module, dict[str, Any]]:
    if detector.kind == "dinomaly":
        return _DinomalyGraph(detector), {"kind": "dinomaly-onnx", "max_ratio": detector.max_ratio}
    if detector.kind == "patchcore":
        graph = _PatchCoreGraph(detector)
        if detector.vit:
            grid = detector.image_size // detector._backbone.patch_size
            grid = (grid, grid)
        else:
            with torch.no_grad():
                _, grid = detector._backbone(torch.zeros(1, 3, detector.image_size, detector.image_size,
                                                         device=detector.device))
        return graph, {"kind": "patchcore-onnx", "vit": bool(detector.vit), "grid": list(grid)}
    raise ValueError(f"ONNX export supports dinomaly and patchcore models, not '{detector.kind}' "
                     "(export ensemble members individually)")


def export_onnx(detector, path: str | Path, *, opset: int = 17, atol: float = 1e-3,
                check_images: Sequence[np.ndarray] | None = None, score_rtol: float = 0.02,
                calibration_images: Sequence[np.ndarray] | None = None) -> dict[str, Any]:
    """Export, verify graph parity, and optionally re-derive the threshold from ONNX scores.

    ``calibration_images`` must be good parts the model was *not* trained on; the threshold
    is then re-computed with the same conformal rule on the exported model's own scores,
    which removes any PyTorch (bf16/fp16) vs ONNX (fp32) numeric gap from the decision.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph, meta = _graph_and_meta(detector)
    graph = graph.float().eval().cpu()
    size = detector.image_size
    dummy = torch.randn(1, 3, size, size)
    torch.onnx.export(
        graph, (dummy,), str(path), opset_version=opset, input_names=["image"], output_names=["anomaly"],
        dynamic_axes={"image": {0: "batch"}, "anomaly": {0: "batch"}}, do_constant_folding=True,
    )
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    probe = torch.randn(2, 3, size, size)
    with torch.no_grad():
        ref = graph(probe).numpy()
    got = session.run(None, {"image": probe.numpy()})[0]
    max_err = float(np.abs(ref - got).max())
    if max_err > atol:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"ONNX parity check failed (max abs error {max_err:.2e} > {atol})")
    sidecar = {
        **meta, "image_size": size, "threshold": detector.threshold, "calibration": detector.calibration,
        "backbone": detector.backbone_name, "parity_max_abs_error": max_err,
        "preprocess": detector.preprocessor.state(),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    if calibration_images:
        from src.inference.anomaly.calibration import conformal_threshold

        served = OnnxDetector(path, providers=["CPUExecutionProvider"])
        cal = conformal_threshold([served.infer(img)[0] for img in calibration_images],
                                  target_frr=float(detector.calibration.get("target_frr", 0.01)),
                                  confidence=detector.calibration.get("confidence"))
        sidecar["threshold"] = float(cal["threshold"])
        sidecar["calibration"] = {**detector.calibration, **cal, "recalibrated_on": "onnx"}
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    if check_images:
        served = OnnxDetector(path, providers=["CPUExecutionProvider"])
        diffs = []
        for img in check_images:
            a = detector.infer(img)[0]
            b = served.infer(img)[0]
            diffs.append(abs(a - b) / max(abs(a), 1e-12))
        sidecar["score_max_rel_diff"] = float(max(diffs))
        if sidecar["score_max_rel_diff"] > score_rtol and not calibration_images:
            path.unlink(missing_ok=True)
            path.with_suffix(".json").unlink(missing_ok=True)
            raise RuntimeError(f"Exported scores differ from PyTorch by {sidecar['score_max_rel_diff']:.2%} "
                               f"(> {score_rtol:.0%}): pass held-out good images as calibration_images "
                               "(--calibration-images) so the threshold matches the exported model")
        path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str), encoding="utf-8")
    return sidecar


def export_dinomaly_onnx(detector, path: str | Path, **kwargs) -> dict[str, Any]:
    """Backwards-compatible name."""
    if detector.kind != "dinomaly":
        raise ValueError("export_dinomaly_onnx expects a Dinomaly model (use export_onnx)")
    return export_onnx(detector, path, **kwargs)


class OnnxDetector(AnomalyDetector):
    """Serve an exported model with ONNX Runtime (no PyTorch model code needed)."""

    def __init__(self, path: str | Path, providers: list[str] | None = None):
        super().__init__("cpu")
        import onnxruntime as ort

        self.path = Path(path)
        meta = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
        self.kind = meta.get("kind", "dinomaly-onnx")
        self.meta = meta
        self.backbone_name = meta.get("backbone")
        self.image_size = int(meta["image_size"])
        self.max_ratio = float(meta.get("max_ratio", 0.01))
        self.threshold = float(meta["threshold"])
        self.calibration = dict(meta.get("calibration") or {})
        self.preprocessor = Preprocessor.from_state(meta.get("preprocess"))
        available = ort.get_available_providers()
        preferred = providers or [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                                  if p in available]
        self.session = ort.InferenceSession(str(self.path), providers=preferred)
        logger.info("[ONNX] %s on %s", self.path.name, self.session.get_providers()[0])

    def _train(self, images, **kwargs):
        raise NotImplementedError("Exported models are inference-only; retrain the PyTorch model")

    def score_map(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        x = to_tensor_batch([image], self.image_size, torch.device("cpu")).numpy()
        out = self.session.run(None, {"image": x})[0][0]
        h, w = image.shape[:2]
        if self.kind == "dinomaly-onnx":
            smooth = cv2.GaussianBlur(out.astype(np.float32), (5, 5), sigmaX=4)
            flat = np.sort(smooth.ravel())[::-1]
            score = float(flat[: max(1, int(flat.size * self.max_ratio))].mean())
            return score, cv2.resize(smooth, (w, h), interpolation=cv2.INTER_LINEAR)
        gh, gw = self.meta["grid"]
        patch = out.reshape(gh, gw).astype(np.float32)
        amap = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
        amap = cv2.GaussianBlur(amap, (0, 0), sigmaX=max(1.0, 4.0 * h / self.image_size))
        if self.meta.get("vit"):
            k = max(1, int(round(0.01 * patch.size)))
            score = float(np.sort(patch.ravel())[::-1][:k].mean())
        else:
            score = float(amap.max())
        return score, amap.astype(np.float32)

    def _state(self):
        raise NotImplementedError

    @classmethod
    def _from_state(cls, state, device):
        raise NotImplementedError


OnnxDinomalyDetector = OnnxDetector  # backwards-compatible name


def main(argv=None) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--check-images", default=None, help="Directory of images to compare PyTorch vs ONNX scores")
    parser.add_argument("--check-limit", type=int, default=20)
    parser.add_argument("--score-rtol", type=float, default=0.02)
    parser.add_argument("--calibration-images", default=None,
                        help="Held-out good images (not used in training) to re-derive the threshold on ONNX scores")
    args = parser.parse_args(argv)
    from src.inference.anomaly.registry import load_detector

    check = None
    if args.check_images:
        from src.inference.anomaly.train import load_images

        check = load_images(args.check_images, args.check_limit)
    calib = None
    if args.calibration_images:
        from src.inference.anomaly.train import load_images

        calib = load_images(args.calibration_images)
    detector = load_detector(args.model, device="cuda" if torch.cuda.is_available() else "cpu")
    meta = export_onnx(detector, args.out, check_images=check, score_rtol=args.score_rtol, calibration_images=calib)
    logger.info("Exported %s (parity max abs error %.2e, score rel diff %s)", args.out,
                meta["parity_max_abs_error"], meta.get("score_max_rel_diff"))
    return meta


if __name__ == "__main__":
    main()
