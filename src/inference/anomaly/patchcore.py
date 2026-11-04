"""PatchCore: nearest-neighbour patch memory of good parts (no training step).

Backbones:

* CNN (``wide_resnet50_2``): the original PatchCore recipe (Roth et al., CVPR 2022) -
  layer2+layer3 features, greedy k-center coreset, Euclidean distance, max-pooled map.
* DINO ViT (``dinov2_vits14``, ``dinov2reg_vitb14``, ``dinov3_*`` ...): the AnomalyDINO
  recipe (Damm et al., WACV 2025) - L2-normalised patch tokens, cosine distance, image
  score = mean of the top 1% patch distances. Strong from very few good images.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.inference.anomaly.backbones import CNNBackbone, ViTBackbone, is_vit, to_tensor_batch
from src.inference.anomaly.base import AnomalyDetector, AnomalyResult  # noqa: F401  (re-export)

logger = logging.getLogger("QualyxusPatchCore")


def greedy_coreset(features: torch.Tensor, fraction: float, *, projection_dim: int = 128,
                   seed: int = 0, max_size: int = 30000, device: torch.device | None = None) -> torch.Tensor:
    """Indices of an approximate k-center coreset (iterative farthest-point selection)."""
    n = features.shape[0]
    k = int(min(max_size, max(1, math.ceil(n * fraction))))
    if k >= n:
        return torch.arange(n, device=features.device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    if features.shape[1] > projection_dim:
        proj = torch.randn(features.shape[1], projection_dim, generator=gen).to(features.device, features.dtype)
        reduced = features @ proj
    else:
        reduced = features
    reduced = reduced.to(device or features.device)  # small enough for the GPU once projected
    selected = torch.empty(k, dtype=torch.long, device=reduced.device)
    current = int(torch.randint(n, (1,), generator=gen).item())
    min_dist = torch.full((n,), float("inf"), device=reduced.device, dtype=reduced.dtype)
    for i in range(k):
        selected[i] = current
        dist = torch.linalg.vector_norm(reduced - reduced[current], dim=1)
        min_dist = torch.minimum(min_dist, dist)
        current = int(torch.argmax(min_dist).item())
    return selected


class PatchCoreDetector(AnomalyDetector):
    kind = "patchcore"

    def __init__(
        self,
        *,
        backbone: str = "wide_resnet50_2",
        image_size: int = 256,
        embed_dim: int = 512,
        coreset_fraction: float = 0.1,
        device: str | None = None,
        vit_layer: int | None = None,
    ):
        super().__init__(device)
        self.backbone_name = backbone
        self.vit = is_vit(backbone)
        self.image_size = int(image_size)
        self.embed_dim = int(embed_dim)
        self.coreset_fraction = float(coreset_fraction)
        self.vit_layer = vit_layer
        self.memory_bank: torch.Tensor | None = None
        self._backbone = None

    # ------------------------------------------------------------- features

    def _ensure_backbone(self):
        if self._backbone is None:
            if self.vit:
                self._backbone = ViTBackbone(self.backbone_name, self.device)
                self.image_size = self._backbone.align_size(self.image_size)
                if self.vit_layer is None:
                    self.vit_layer = self._backbone.num_layers - 1
            else:
                self._backbone = CNNBackbone(self.backbone_name, self.device)
        return self._backbone

    @torch.inference_mode()
    def _patch_features(self, images: Sequence[np.ndarray]) -> tuple[torch.Tensor, tuple[int, int]]:
        backbone = self._ensure_backbone()
        x = to_tensor_batch(images, self.image_size, self.device)
        if self.vit:
            tokens = backbone(x, [self.vit_layer])
            feats = F.normalize(tokens.layers[0].reshape(-1, tokens.layers[0].shape[-1]), dim=-1)
            return feats, tokens.grid
        feats, grid = backbone(x)
        feats = feats.reshape(-1, feats.shape[-1])
        feats = F.adaptive_avg_pool1d(feats.unsqueeze(1), self.embed_dim).squeeze(1)
        return feats.float(), grid

    # -------------------------------------------------------------- training

    def _train(self, images: Sequence[np.ndarray], batch_size: int = 16, seed: int = 0, **_) -> dict[str, Any]:
        chunks = []
        for start in range(0, len(images), batch_size):
            feats, _ = self._patch_features(images[start:start + batch_size])
            chunks.append(feats.cpu())  # the full bank is far larger than the final model
        bank = torch.cat(chunks, dim=0)
        finite = torch.isfinite(bank).all(dim=1)
        if not bool(finite.all()):
            logger.warning("[PatchCore] Dropping %d non-finite patch features", int((~finite).sum()))
            bank = bank[finite]
        idx = greedy_coreset(bank, self.coreset_fraction, seed=seed, device=self.device).to(bank.device)
        # Round to the stored fp16 precision now, so calibration sees exactly the deployed bank.
        self.memory_bank = bank[idx].half().float().contiguous().to(self.device)
        logger.info("[PatchCore/%s] Memory bank: %d patches (from %d) x %d dims",
                    self.backbone_name, self.memory_bank.shape[0], bank.shape[0], self.memory_bank.shape[1])
        return {"memory_bank": int(self.memory_bank.shape[0]), "backbone": self.backbone_name}

    # ------------------------------------------------------------- inference

    @torch.inference_mode()
    def score_map(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore detector is not fitted/loaded")
        feats, (h, w) = self._patch_features([image])
        best = None
        for start in range(0, self.memory_bank.shape[0], 8192):
            bank = self.memory_bank[start:start + 8192]
            if self.vit:
                d = 1.0 - feats @ bank.T  # cosine distance on unit vectors
            else:
                d = torch.cdist(feats, bank)
            chunk_min = d.min(dim=1).values
            best = chunk_min if best is None else torch.minimum(best, chunk_min)
        patch_scores = best
        patch_map = patch_scores.view(1, 1, h, w)
        full = F.interpolate(patch_map, size=image.shape[:2], mode="bilinear", align_corners=False)[0, 0]
        amap = full.float().cpu().numpy()
        sigma = max(1.0, 4.0 * image.shape[0] / self.image_size)
        amap = cv2.GaussianBlur(amap, (0, 0), sigmaX=sigma)
        if self.vit:
            k = max(1, int(round(0.01 * patch_scores.numel())))
            score = float(torch.topk(patch_scores, k).values.mean())
        else:
            score = float(amap.max())
        return score, amap.astype(np.float32)

    # ----------------------------------------------------------- persistence

    def _state(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "image_size": self.image_size,
            "embed_dim": self.embed_dim,
            "coreset_fraction": self.coreset_fraction,
            "vit_layer": self.vit_layer,
            "memory_bank": self.memory_bank.half().cpu(),
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any], device: str | None) -> "PatchCoreDetector":
        det = cls(backbone=state["backbone"], image_size=state["image_size"], embed_dim=state["embed_dim"],
                  coreset_fraction=state["coreset_fraction"], device=device, vit_layer=state.get("vit_layer"))
        det.memory_bank = state["memory_bank"].float().to(det.device)
        return det

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "PatchCoreDetector":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") == "qualyxus-patchcore-v1":  # legacy single-file format
            det = cls(backbone=payload["backbone"], image_size=payload["image_size"], embed_dim=payload["embed_dim"],
                      coreset_fraction=payload["coreset_fraction"], device=device)
            det.memory_bank = payload["memory_bank"].float().to(det.device)
            det.threshold = float(payload["threshold"])
            det.calibration = dict(payload.get("calibration") or {})
            return det
        if payload.get("kind") != cls.kind:
            raise ValueError(f"{path} is not a PatchCore model")
        det = cls._from_state(payload["state"], device)
        det._restore_common(payload)
        return det
