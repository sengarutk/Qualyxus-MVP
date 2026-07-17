from __future__ import annotations

import numpy as np
import cv2
import torch
from typing import Any

# Ultralytics/YOLO models expect 0..1 scaled RGB by default.
YOLO_RGB_MEAN = (0.0, 0.0, 0.0)
YOLO_RGB_STD = (1.0, 1.0, 1.0)

METADATA_KEY_ALIASES: dict[str, str] = {
    "lineId": "line_id",
    "stationId": "station_id",
    "auditHash": "audit_hash",
    "machineId": "machine_id",
    "shiftId": "shift_id",
    "batchId": "batch_id",
    "operatorId": "operator_id",
    "lotId": "lot_id",
    "fullImagePath": "full_image_path",
    "overlayPath": "overlay_path",
    "cropPath": "crop_path",
    "defectType": "defect_type",
}

def normalize_metadata_keys(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    out: dict[str, Any] = {}
    for key, value in dict(metadata).items():
        normalized = METADATA_KEY_ALIASES.get(str(key), str(key))
        out[normalized] = value
    return out

class UnifiedPreprocessor:
    """
    Centralized, hardware-aware tensor allocation preprocessor.
    Resolves PCIe stalls and pre-allocates normalization arrays.
    """
    def __init__(self, use_triton: bool = True, precision: str = "fp16"):
        self.use_triton = use_triton
        self.precision = str(precision).lower().strip()
        self.triton = None
        if use_triton and torch.cuda.is_available():
            try:
                from src.inference.kernels.triton_ops import TritonPreprocessor
                self.triton = TritonPreprocessor(precision=precision)
            except (ImportError, RuntimeError):
                pass
                
        # Flaw Fix (Issue 9): Indexed lookup slots to prevent cross-camera thread data corruption
        self._cpu_mean_cache: dict[str, np.ndarray] = {}
        self._cpu_std_cache: dict[str, np.ndarray] = {}
        self._gpu_mean_cache: dict[str, torch.Tensor] = {}
        self._gpu_std_cache: dict[str, torch.Tensor] = {}
        
        # Hardening Fix: Pre-allocated persistent pinned memory pool 
        # Prevents CPU thrashing caused by dynamic OS page-locking per frame.
        self._pinned_buffers: dict[str, torch.Tensor] = {}

    def _get_pinned_buffer(self, camera_id: str, shape: tuple) -> torch.Tensor:
        if camera_id not in self._pinned_buffers or self._pinned_buffers[camera_id].shape != shape:
            self._pinned_buffers[camera_id] = torch.empty(shape, dtype=torch.uint8, pin_memory=True)
        return self._pinned_buffers[camera_id]

    def _get_cpu_arrays(self, camera_id: str, mean: tuple, std: tuple):
        if camera_id not in self._cpu_mean_cache:
            self._cpu_mean_cache[camera_id] = np.array(mean, dtype=np.float32)
            self._cpu_std_cache[camera_id] = np.array(std, dtype=np.float32)
        return self._cpu_mean_cache[camera_id], self._cpu_std_cache[camera_id]

    def _get_gpu_tensors(self, camera_id: str, mean: tuple, std: tuple, out: torch.Tensor):
        if camera_id not in self._gpu_mean_cache:
            target_dtype = torch.float16 if self.precision == "fp16" else torch.float32
            self._gpu_mean_cache[camera_id] = torch.tensor(mean, device=out.device, dtype=target_dtype).view(1, 1, 3)
            self._gpu_std_cache[camera_id] = torch.tensor(std, device=out.device, dtype=target_dtype).view(1, 1, 3)
        return self._gpu_mean_cache[camera_id], self._gpu_std_cache[camera_id]

    def process(self, frame: np.ndarray, device: str = "cuda", camera_id: str = "default", mean=YOLO_RGB_MEAN, std=YOLO_RGB_STD):
        # Only utilize pinned memory transfers if targeting GPU and source is standard numpy array
        if device == "cuda" and isinstance(frame, np.ndarray):
            tensor_src = self._get_pinned_buffer(camera_id, frame.shape)
            tensor_src.copy_(torch.from_numpy(frame))
        else:
            tensor_src = frame

        if self.triton is not None and device == "cuda":
            return self.triton.process(tensor_src, device=device)
            
        if device == "cuda":
            return self.gpu_normalize_frame(tensor_src, camera_id, mean, std)
        return self.cpu_normalize_frame(tensor_src, camera_id, mean, std)

    def gpu_normalize_frame(self, frame: np.ndarray | torch.Tensor, camera_id: str = "default", mean: tuple = YOLO_RGB_MEAN, std: tuple = YOLO_RGB_STD):
        """GPU-side cast + normalize utilizing true page-locked pinned host allocations."""
        if isinstance(frame, np.ndarray):
            src = torch.from_numpy(frame)
            if not src.is_contiguous():
                src = src.contiguous()
            src_pinned = self._get_pinned_buffer(camera_id, src.shape)
            src_pinned.copy_(src)
        else:
            src_pinned = frame
            
        src_gpu = src_pinned.to(device="cuda", dtype=torch.uint8, non_blocking=True).contiguous()
        
        # Hardening Fix: Convert to target precision early to decrease register pressure and VRAM footprint
        target_dtype = torch.float16 if self.precision == "fp16" else torch.float32
        out = src_gpu.to(dtype=target_dtype)
        
        # Flaw Fix (Issue 7): Eliminate .item() PCIe stall. Apply strict mathematical bounds asynchronously.
        out.div_(255.0)

        out = out[..., [2, 1, 0]].contiguous(memory_format=torch.contiguous_format)
        if mean != YOLO_RGB_MEAN or std != YOLO_RGB_STD:
            mean_t, std_t = self._get_gpu_tensors(camera_id, mean, std, out)
            out = (out - mean_t) / std_t

        out = out.permute(2, 0, 1).contiguous(memory_format=torch.contiguous_format).unsqueeze(0)
        return out

    def cpu_normalize_frame(self, frame: np.ndarray | torch.Tensor, camera_id: str = "default", mean: tuple = YOLO_RGB_MEAN, std: tuple = YOLO_RGB_STD):
        """Hardened CPU normalizer. Guarantees float32 allocations to prevent fallback matrix crashes."""
        if isinstance(frame, torch.Tensor):
            frame = frame.numpy()
            
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        arr = rgb.astype(np.float32) / 255.0
        if mean != YOLO_RGB_MEAN or std != YOLO_RGB_STD:
            mean_arr, std_arr = self._get_cpu_arrays(camera_id, mean, std)
            arr = (arr - mean_arr) / std_arr
        arr = np.transpose(arr, (2, 0, 1))
        # Enforce float32 output for CPU pipelines to avoid unsupported half-precision operations
        return torch.from_numpy(arr).unsqueeze(0).float()

def make_preprocessor(use_triton: bool = True, precision: str = "fp16"):
    return UnifiedPreprocessor(use_triton=use_triton, precision=precision)

def preprocess_for_inference(
    frame: np.ndarray,
    *,
    device: str,
    use_triton: bool,
    precision: str = "fp16",
    preprocessor=None,
):
    if preprocessor is not None:
        return preprocessor.process(frame, device=device), True if hasattr(preprocessor, "triton") and preprocessor.triton is not None else False
    
    # Fallback to ephemeral preprocessor creation
    p = UnifiedPreprocessor(use_triton=use_triton, precision=precision)
    return p.process(frame, device=device), True if p.triton is not None else False

def overlay_mask(image, mask, alpha=0.5, color=(0, 0, 255)):
    h, w = image.shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[:] = color
    m = mask.astype(bool)
    overlay[m] = cv2.addWeighted(image[m], 1 - alpha, colored[m], alpha, 0)
    return overlay

def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    return int(x1), int(y1), int(x2), int(y2)

def severity_from_conf(conf: float) -> int:
    if conf >= 0.85: return 3
    if conf >= 0.6: return 2
    return 1
