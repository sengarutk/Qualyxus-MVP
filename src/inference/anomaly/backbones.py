"""Frozen feature backbones shared by the anomaly detectors.

Two families:

* CNN (``wide_resnet50_2``, ``resnet18``; torchvision, ImageNet weights, BSD license):
  layer2 + layer3 feature maps - the classic PatchCore setting.
* Vision transformers from the DINO self-supervised family (Hugging Face weights):

  ============================  =========================================  ==============
  name                          checkpoint                                 license
  ============================  =========================================  ==============
  ``dinov2_vits14``             facebook/dinov2-small                      Apache-2.0
  ``dinov2_vitb14``             facebook/dinov2-base                       Apache-2.0
  ``dinov2_vitl14``             facebook/dinov2-large                      Apache-2.0
  ``dinov2reg_vits14``          facebook/dinov2-with-registers-small       Apache-2.0
  ``dinov2reg_vitb14``          facebook/dinov2-with-registers-base        Apache-2.0
  ``dinov2reg_vitl14``          facebook/dinov2-with-registers-large       Apache-2.0
  ``dinov3_vits16`` / ``vitb16`` / ``vitl16``  facebook/dinov3-*-pretrain-lvd1689m  DINOv3 License
  ============================  =========================================  ==============

DINOv3 weights are gated on Hugging Face: accept Meta's license with your own account and
export ``HF_TOKEN`` before selecting them. They are opt-in because the license is custom.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("QualyxusBackbones")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

VIT_CHECKPOINTS: dict[str, tuple[str, str]] = {
    "dinov2_vits14": ("facebook/dinov2-small", "Apache-2.0"),
    "dinov2_vitb14": ("facebook/dinov2-base", "Apache-2.0"),
    "dinov2_vitl14": ("facebook/dinov2-large", "Apache-2.0"),
    "dinov2reg_vits14": ("facebook/dinov2-with-registers-small", "Apache-2.0"),
    "dinov2reg_vitb14": ("facebook/dinov2-with-registers-base", "Apache-2.0"),
    "dinov2reg_vitl14": ("facebook/dinov2-with-registers-large", "Apache-2.0"),
    "dinov3_vits16": ("facebook/dinov3-vits16-pretrain-lvd1689m", "DINOv3-License"),
    "dinov3_vitb16": ("facebook/dinov3-vitb16-pretrain-lvd1689m", "DINOv3-License"),
    "dinov3_vitl16": ("facebook/dinov3-vitl16-pretrain-lvd1689m", "DINOv3-License"),
}
CNN_BACKBONES = ("wide_resnet50_2", "resnet18")


def is_vit(name: str) -> bool:
    return name in VIT_CHECKPOINTS


def backbone_license(name: str) -> str:
    if "+" in name:  # ensemble: every member's licence applies
        return " + ".join(dict.fromkeys(backbone_license(n) for n in name.split("+")))
    if name in CNN_BACKBONES:
        return "BSD-3-Clause (torchvision)"
    return VIT_CHECKPOINTS[name][1]


def autocast_dtype(device: torch.device) -> torch.dtype:
    """bfloat16 where supported (same range as fp32, no overflow); fp16 otherwise."""
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_tensor_batch(images: Sequence[np.ndarray], size: int, device: torch.device) -> torch.Tensor:
    """BGR uint8 images -> ImageNet-normalised NCHW float tensor at ``size`` x ``size``."""
    batch = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_AREA if min(rgb.shape[:2]) > size else cv2.INTER_LINEAR
        rgb = cv2.resize(rgb, (size, size), interpolation=interp)
        batch.append(torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1))
    x = torch.stack(batch).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device, non_blocking=True)


@dataclass
class ViTTokens:
    layers: list[torch.Tensor]   # each (B, N_patch, C)
    grid: tuple[int, int]        # patch grid (h, w)


class ViTBackbone(torch.nn.Module):
    """Frozen DINO ViT returning per-layer patch tokens (CLS/register tokens removed)."""

    def __init__(self, name: str, device: torch.device | None = None):
        super().__init__()
        if name not in VIT_CHECKPOINTS:
            raise ValueError(f"Unknown ViT backbone '{name}' (choose from {sorted(VIT_CHECKPOINTS)})")
        from transformers import AutoModel

        checkpoint, license_name = VIT_CHECKPOINTS[name]
        if license_name != "Apache-2.0":
            logger.warning("[Backbone] %s is distributed under %s - verify it permits your use.", name, license_name)
        self.name = name
        self.model = AutoModel.from_pretrained(checkpoint).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        cfg = self.model.config
        self.patch_size = int(cfg.patch_size)
        self.embed_dim = int(cfg.hidden_size)
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0) or 0)
        self.device = device or default_device()
        self.model.to(self.device)

    def align_size(self, size: int) -> int:
        return max(self.patch_size, int(round(size / self.patch_size)) * self.patch_size)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, layer_indices: Sequence[int]) -> ViTTokens:
        """``layer_indices`` are 0-based transformer block indices (output of block i)."""
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype(self.device),
                            enabled=self.device.type == "cuda"):
            out = self.model(pixel_values=x, output_hidden_states=True)
        hidden = out.hidden_states  # [embeddings, block0, block1, ...]
        h, w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        layers = [hidden[i + 1][:, self.num_prefix:, :].float() for i in layer_indices]
        if not all(torch.isfinite(t).all() for t in layers):
            # Rare half-precision overflow: recompute this batch in full precision.
            out = self.model(pixel_values=x.float(), output_hidden_states=True)
            layers = [out.hidden_states[i + 1][:, self.num_prefix:, :].float() for i in layer_indices]
        return ViTTokens(layers=layers, grid=(h, w))


class CNNBackbone(torch.nn.Module):
    """Frozen torchvision ResNet returning aggregated layer2+layer3 patch features."""

    def __init__(self, name: str, device: torch.device | None = None):
        super().__init__()
        import torchvision

        if name == "wide_resnet50_2":
            model = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif name == "resnet18":
            model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown CNN backbone '{name}' (choose from {CNN_BACKBONES})")
        self.name = name
        self.device = device or default_device()
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        m = self.model
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        f2 = m.layer2(m.layer1(x))
        f3 = m.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, stride=1, padding=1)
        f3 = F.avg_pool2d(f3, 3, stride=1, padding=1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feats = torch.cat([f2, f3], dim=1)
        b, c, h, w = feats.shape
        return feats.permute(0, 2, 3, 1).reshape(b, h * w, c), (h, w)
