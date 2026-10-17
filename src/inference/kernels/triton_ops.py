from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING, TypeAlias

import numpy as np
import torch
import logging

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None


if TYPE_CHECKING:
    TritonConstexpr: TypeAlias = Any
else:
    TritonConstexpr = tl.constexpr if tl is not None else Any


def _as_float_tuple(values: tuple[float, float, float] | list[float]) -> tuple[float, float, float]:
    vals = tuple(float(v) for v in values)
    if len(vals) != 3:
        raise ValueError("Expected 3-channel mean/std tuple")
    return vals


def _normalize_precision(precision: str) -> str:
    normalized = str(precision).lower().strip()
    if normalized in {"fp16", "float16", "half"}:
        return "fp16"
    if normalized in {"fp32", "float32", "full"}:
        return "fp32"
    if normalized in {"int8", "i8"}:
        # Triton preprocessing still emits normalized floating tensors for model safety.
        # We conservatively map int8 requests to fp16 so downstream inference remains valid.
        return "fp16"
    raise ValueError(f"Unsupported Triton precision: {precision}")


if triton is not None and tl is not None:
    @triton.jit
    def _hwc_bgr_to_chw_norm_fp16_kernel(
        src_ptr,
        dst_ptr,
        height,
        width,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = height * width
        mask = offsets < total

        pixel_idx = offsets
        h = pixel_idx // width
        w = pixel_idx % width
        base = (h * width + w) * 3

        # Source is BGR HWC uint8. Write RGB CHW fp16.
        b = tl.load(src_ptr + base + 0, mask=mask, other=0).to(tl.float32) / 255.0
        g = tl.load(src_ptr + base + 1, mask=mask, other=0).to(tl.float32) / 255.0
        r = tl.load(src_ptr + base + 2, mask=mask, other=0).to(tl.float32) / 255.0

        r = (r - mean0) / std0
        g = (g - mean1) / std1
        b = (b - mean2) / std2

        plane = height * width
        tl.store(dst_ptr + pixel_idx, r.to(tl.float16), mask=mask)
        tl.store(dst_ptr + plane + pixel_idx, g.to(tl.float16), mask=mask)
        tl.store(dst_ptr + (2 * plane) + pixel_idx, b.to(tl.float16), mask=mask)


if triton is not None and tl is not None:
    @triton.jit
    def _hwc_bgr_to_chw_norm_fp32_kernel(
        src_ptr,
        dst_ptr,
        height,
        width,
        mean0,
        mean1,
        mean2,
        std0,
        std1,
        std2,
        BLOCK_SIZE: tl.constexpr,  # pyright: ignore[reportInvalidTypeForm]
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = height * width
        mask = offsets < total

        pixel_idx = offsets
        h = pixel_idx // width
        w = pixel_idx % width
        base = (h * width + w) * 3

        b = tl.load(src_ptr + base + 0, mask=mask, other=0).to(tl.float32) / 255.0
        g = tl.load(src_ptr + base + 1, mask=mask, other=0).to(tl.float32) / 255.0
        r = tl.load(src_ptr + base + 2, mask=mask, other=0).to(tl.float32) / 255.0

        r = (r - mean0) / std0
        g = (g - mean1) / std1
        b = (b - mean2) / std2

        plane = height * width
        tl.store(dst_ptr + pixel_idx, r, mask=mask)
        tl.store(dst_ptr + plane + pixel_idx, g, mask=mask)
        tl.store(dst_ptr + (2 * plane) + pixel_idx, b, mask=mask)


@dataclass
class TritonKernelConfig:
    block_size: int = 1024
    # Ultralytics/YOLO expects inputs scaled to 0..1; it does NOT expect Imagenet mean/std.
    # If you apply Imagenet normalization here, Ultralytics may incorrectly divide by 255 again.
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    precision: str = "fp16"


class TritonPreprocessor:
    """Adaptive Triton preprocessing with graceful fallback hooks."""

    def __init__(
        self,
        block_size: int = 1024,
        mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
        std: tuple[float, float, float] = (1.0, 1.0, 1.0),
        precision: str = "fp16",
    ):
        if triton is None or tl is None:
            raise RuntimeError("Triton is not available")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        self.cfg = TritonKernelConfig(
            block_size=int(block_size),
            mean=_as_float_tuple(mean),
            std=_as_float_tuple(std),
            precision=_normalize_precision(precision),
        )

    @property
    def is_available(self) -> bool:
        return triton is not None and torch.cuda.is_available()

    def _grid(self, height: int, width: int):
        total = int(height) * int(width)
        return (triton.cdiv(total, self.cfg.block_size),)

    def process(self, frame: np.ndarray | torch.Tensor, device: str = "cuda") -> torch.Tensor:
        if not self.is_available:
            raise RuntimeError("Triton preprocessing unavailable")

        # Fast path: caller already provided a normalized (1,3,H,W) GPU tensor.
        if isinstance(frame, torch.Tensor) and frame.is_cuda and frame.ndim == 4:
            if int(frame.shape[0]) == 1 and int(frame.shape[1]) == 3:
                dtype = torch.float16 if self.cfg.precision == "fp16" else torch.float32
                return frame.to(dtype=dtype).contiguous(memory_format=torch.contiguous_format)
            raise ValueError("Expected a (1,3,H,W) tensor for pre-normalized GPU input")

        # Raw HWC BGR uint8 input (host or device) always goes through the kernel so the
        # output is RGB, CHW and scaled to 0..1 regardless of where the frame lives.
        if isinstance(frame, np.ndarray):
            src = torch.from_numpy(frame).contiguous()
        else:
            src = frame.detach().contiguous()

        if src.ndim != 3 or int(src.shape[2]) != 3:
            raise ValueError("Expected HWC image with 3 channels")

        height, width = int(src.shape[0]), int(src.shape[1])
        src_gpu = src.to(device=device, dtype=torch.uint8, non_blocking=True).contiguous()
        
        # Flaw Fix (Issue 8): Dynamically evaluate dtype allocation instead of statically mapping to fp32
        dst_dtype = torch.float16 if self.cfg.precision == "fp16" else torch.float32
        dst = torch.empty((3, height, width), device=device, dtype=dst_dtype).contiguous(memory_format=torch.contiguous_format)

        try:
            if self.cfg.precision == "fp16":
                _hwc_bgr_to_chw_norm_fp16_kernel[self._grid(height, width)](
                    src_gpu,
                    dst,
                    height,
                    width,
                    self.cfg.mean[0],
                    self.cfg.mean[1],
                    self.cfg.mean[2],
                    self.cfg.std[0],
                    self.cfg.std[1],
                    self.cfg.std[2],
                    BLOCK_SIZE=self.cfg.block_size,
                )
            else:
                _hwc_bgr_to_chw_norm_fp32_kernel[self._grid(height, width)](
                    src_gpu,
                    dst,
                    height,
                    width,
                    self.cfg.mean[0],
                    self.cfg.mean[1],
                    self.cfg.mean[2],
                    self.cfg.std[0],
                    self.cfg.std[1],
                    self.cfg.std[2],
                    BLOCK_SIZE=self.cfg.block_size,
                )
            return dst.contiguous(memory_format=torch.contiguous_format).unsqueeze(0)
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(f"Triton OOM: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Triton kernel failed: {exc}") from exc

    def benchmark_name(self) -> str:
        return "triton_kernel"