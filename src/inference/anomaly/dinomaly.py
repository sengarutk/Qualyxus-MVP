"""Dinomaly (Guo et al., CVPR 2025): reconstruction of frozen DINO features.

A frozen DINOv2-with-registers encoder produces multi-layer patch features. A lightweight
decoder learns to reconstruct them from a deliberately *lossy* input:

* noisy bottleneck - an MLP with dropout 0.2 on the fused encoder features, so the decoder
  cannot simply copy its input (no "identity shortcut"),
* linear attention (ELU+1 kernel) - unfocused attention that again prevents copying,
* loose reconstruction - layers are fused into two groups and compared with a *global*
  cosine loss, with hard-point mining: gradients of the easiest (1 - p) fraction of points
  are scaled down so training focuses on what is still poorly reconstructed.

At test time anomalies are regions the decoder, trained only on good parts, fails to
reconstruct: anomaly map = 1 - cosine(encoder group, decoder group), averaged over groups.

Defaults follow the official implementation (ViT-B/14-reg, layers 2..9, groups
[[0..3],[4..7]], StableAdamW-like AdamW lr 2e-3 -> 2e-4 warm-cosine, grad-clip 0.1,
image score = mean of the top 1% of the map), with batch size and iteration count
reduced for single-recipe training on a 6 GB GPU.
"""
from __future__ import annotations

import logging
import math
import time
from functools import partial
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.inference.anomaly.backbones import ViTBackbone, autocast_dtype, to_tensor_batch
from src.inference.anomaly.base import AnomalyDetector

logger = logging.getLogger("QualyxusDinomaly")


class BottleneckMLP(nn.Module):
    """Noisy bottleneck: dropout on input, hidden activation and output (as in bMlp)."""

    def __init__(self, dim: int, hidden: int, drop: float = 0.2):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, hidden), nn.Linear(hidden, dim)
        self.act, self.drop = nn.GELU(), nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(self.drop(x))))))


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = F.elu(qkv[0]) + 1.0, F.elu(qkv[1]) + 1.0, qkv[2]
        kv = torch.einsum("bhsd,bhse->bhde", k, v)
        z = 1.0 / (torch.einsum("bhsd,bhd->bhs", q, k.sum(dim=-2)) + 1e-6)
        out = torch.einsum("bhde,bhsd,bhs->bhse", kv, q, z)
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        norm = partial(nn.LayerNorm, eps=1e-8)
        self.norm1, self.attn = norm(dim), LinearAttention(dim, heads)
        self.norm2 = norm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class DinomalyHead(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int = 8, drop: float = 0.2):
        super().__init__()
        self.bottleneck = BottleneckMLP(dim, dim * 4, drop)
        self.blocks = nn.ModuleList([DecoderBlock(dim, heads) for _ in range(depth)])

    def forward(self, fused: torch.Tensor) -> list[torch.Tensor]:
        x = self.bottleneck(fused)
        outs = []
        for blk in self.blocks:
            x = blk(x)
            outs.append(x)
        return outs[::-1]  # reversed so decoder group i mirrors encoder group i


def _global_cosine_hm(en: list[torch.Tensor], de: list[torch.Tensor], p: float, factor: float) -> torch.Tensor:
    """Global cosine loss with hard-point mining (``global_cosine_hm_percent``)."""
    loss = 0.0
    for a, b in zip(en, de):  # (B, C, H, W)
        a = a.detach()
        with torch.no_grad():
            point_dist = 1.0 - F.cosine_similarity(a, b, dim=1).unsqueeze(1)
            k = max(1, int(point_dist.numel() * (1.0 - p)))
            thresh = torch.topk(point_dist.reshape(-1), k=k).values[-1]
            easy = point_dist < thresh
        loss = loss + torch.mean(1.0 - F.cosine_similarity(a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1), dim=1))

        def _scale(grad, mask=easy):
            grad = grad.clone()
            grad[mask.expand_as(grad)] *= factor
            return grad

        if b.requires_grad:
            b.register_hook(_scale)
    return loss / len(en)


class DinomalyDetector(AnomalyDetector):
    kind = "dinomaly"

    def __init__(
        self,
        *,
        backbone: str = "dinov2reg_vitb14",
        image_size: int = 392,
        target_layers: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9),
        groups: Sequence[Sequence[int]] = ((0, 1, 2, 3), (4, 5, 6, 7)),
        device: str | None = None,
        max_ratio: float = 0.01,
    ):
        super().__init__(device)
        self.backbone_name = backbone
        self.image_size = int(image_size)
        self.target_layers = [int(i) for i in target_layers]
        self.groups = [list(g) for g in groups]
        self.max_ratio = float(max_ratio)
        self._backbone: ViTBackbone | None = None
        self.head: DinomalyHead | None = None

    def _ensure(self) -> ViTBackbone:
        if self._backbone is None:
            self._backbone = ViTBackbone(self.backbone_name, self.device)
            self.image_size = self._backbone.align_size(self.image_size)
            if max(self.target_layers) >= self._backbone.num_layers:
                raise ValueError(f"target_layers exceed the {self._backbone.num_layers} blocks of {self.backbone_name}")
        if self.head is None:
            dim = self._backbone.embed_dim
            self.head = DinomalyHead(dim, heads=max(1, dim // 64), depth=len(self.target_layers)).to(self.device)
        return self._backbone

    def _features(self, x: torch.Tensor):
        backbone = self._ensure()
        tokens = backbone(x, self.target_layers)
        fused = torch.stack(tokens.layers, dim=1).mean(dim=1)
        return tokens, fused

    def _group_maps(self, layers: list[torch.Tensor], grid: tuple[int, int]) -> list[torch.Tensor]:
        h, w = grid
        out = []
        for g in self.groups:
            t = torch.stack([layers[i] for i in g], dim=1).mean(dim=1)  # B, N, C
            out.append(t.permute(0, 2, 1).reshape(t.shape[0], t.shape[2], h, w).contiguous())
        return out

    # -------------------------------------------------------------- training

    def _train(self, images: Sequence[np.ndarray], *, iterations: int = 3000, batch_size: int = 8,
               lr: float = 2e-3, final_lr: float = 2e-4, warmup: int = 100, p_final: float = 0.9,
               p_ramp: int = 1000, hard_factor: float = 0.1, seed: int = 0, log_every: int = 500, **_) -> dict[str, Any]:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self._ensure()
        head = self.head
        head.train()
        optim = torch.optim.AdamW(head.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4,
                                  amsgrad=True, eps=1e-10)
        scaler = torch.amp.GradScaler(
            enabled=self.device.type == "cuda" and autocast_dtype(self.device) == torch.float16)
        # Pre-tensorise once: training images are re-used every iteration.
        bank = to_tensor_batch(images, self.image_size, torch.device("cpu")).half()  # halves host RAM
        t0, losses = time.time(), []
        for it in range(iterations):
            if it < warmup:
                cur_lr = lr * (it + 1) / warmup
            else:
                progress = (it - warmup) / max(1, iterations - warmup)
                cur_lr = final_lr + 0.5 * (lr - final_lr) * (1 + math.cos(math.pi * progress))
            for group in optim.param_groups:
                group["lr"] = cur_lr
            idx = rng.choice(len(bank), size=min(batch_size, len(bank)), replace=len(bank) < batch_size)
            x = bank[idx].to(self.device, non_blocking=True).float()
            if rng.random() < 0.5:  # cheap photometric jitter for robustness to lighting drift
                x = x * float(rng.uniform(0.9, 1.1)) + float(rng.uniform(-0.1, 0.1))
            tokens, fused = self._features(x)
            with torch.autocast(device_type=self.device.type, dtype=autocast_dtype(self.device),
                                enabled=self.device.type == "cuda"):
                decoded = head(fused)
            en = self._group_maps(tokens.layers, tokens.grid)
            de = self._group_maps([d.float() for d in decoded], tokens.grid)
            p = min(p_final * it / max(p_ramp, 1), p_final)
            loss = _global_cosine_hm(en, de, p=p, factor=hard_factor)
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=0.1)
            scaler.step(optim)
            scaler.update()
            losses.append(float(loss.detach()))
            if log_every and (it + 1) % log_every == 0:
                logger.info("[Dinomaly] iter %d/%d loss %.4f lr %.2e (%.0fs)", it + 1, iterations,
                            float(np.mean(losses[-log_every:])), cur_lr, time.time() - t0)
        head.eval()
        return {"iterations": iterations, "final_loss": float(np.mean(losses[-50:])),
                "train_seconds": round(time.time() - t0, 1), "backbone": self.backbone_name}

    # ------------------------------------------------------------- inference

    @torch.inference_mode()
    def _maps(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        x = to_tensor_batch(images, self.image_size, self.device)
        tokens, fused = self._features(x)
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype(self.device),
                            enabled=self.device.type == "cuda"):
            decoded = self.head(fused)
        en = self._group_maps(tokens.layers, tokens.grid)
        de = self._group_maps([d.float() for d in decoded], tokens.grid)
        maps = [1.0 - F.cosine_similarity(a, b, dim=1).unsqueeze(1) for a, b in zip(en, de)]
        maps = [F.interpolate(m, size=(256, 256), mode="bilinear", align_corners=True) for m in maps]
        return torch.cat(maps, dim=1).mean(dim=1)  # B, 256, 256

    def _finalise(self, amap256: np.ndarray, shape: tuple[int, int]) -> tuple[float, np.ndarray]:
        smooth = cv2.GaussianBlur(amap256, (5, 5), sigmaX=4)
        flat = np.sort(smooth.ravel())[::-1]
        k = max(1, int(flat.size * self.max_ratio))
        score = float(flat[:k].mean())
        full = cv2.resize(smooth, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return score, full.astype(np.float32)

    def score_map(self, image: np.ndarray) -> tuple[float, np.ndarray]:
        if self.head is None:
            raise RuntimeError("Dinomaly detector is not trained/loaded")
        amap = self._maps([image])[0].float().cpu().numpy()
        return self._finalise(amap, image.shape[:2])

    def score_batch(self, images: Sequence[np.ndarray], batch_size: int = 8) -> list[tuple[float, np.ndarray]]:
        out = []
        for start in range(0, len(images), batch_size):
            chunk = images[start:start + batch_size]
            maps = self._maps(chunk).float().cpu().numpy()
            out.extend(self._finalise(m, img.shape[:2]) for m, img in zip(maps, chunk))
        return out

    # ----------------------------------------------------------- persistence

    def _state(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone_name,
            "image_size": self.image_size,
            "target_layers": list(self.target_layers),
            "groups": [list(g) for g in self.groups],
            "max_ratio": self.max_ratio,
            "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()},
        }

    @classmethod
    def _from_state(cls, state: dict[str, Any], device: str | None) -> "DinomalyDetector":
        det = cls(backbone=state["backbone"], image_size=state["image_size"], target_layers=state["target_layers"],
                  groups=state["groups"], device=device, max_ratio=state["max_ratio"])
        det._ensure()
        det.head.load_state_dict(state["head"])
        det.head.eval()
        return det
