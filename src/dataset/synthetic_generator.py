from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.config.config import Config
from src.dataset.yolo_convert import mask_to_polygons, polygon_to_yolo_seg_line


@dataclass
class SyntheticGenerationConfig:
    model_type: str = "inpainting"
    device: str = "cuda"
    batch_size: int = 4
    output_resolution: tuple[int, int] = (640, 640)


class DefectGenerator:
    """Offline synthetic defect data factory for sim-to-real dataset expansion."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config.load()
        raw = self.config.get("synthetic_generation", default={}) or {}
        resolution = raw.get("output_resolution", [640, 640]) or [640, 640]
        self.cfg = SyntheticGenerationConfig(
            model_type=str(raw.get("model_type", "inpainting")),
            device=str(raw.get("device", "cuda")),
            batch_size=int(raw.get("batch_size", 4)),
            output_resolution=(int(resolution[0]), int(resolution[1])),
        )

    def _gpu_has_headroom(self, min_free_gb: float = 4.0) -> bool:
        try:
            import torch

            if self.cfg.device != "cuda" or not torch.cuda.is_available():
                return False
            free_mem, _ = torch.cuda.mem_get_info()
            return free_mem >= int(min_free_gb * 1024**3)
        except Exception:
            return False

    @staticmethod
    def _load_images(directory: Path) -> list[Path]:
        paths: list[Path] = []
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            paths.extend(sorted(directory.glob(pattern)))
        return paths

    @staticmethod
    def _random_blob_mask(height: int, width: int) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (random.randint(width // 5, (4 * width) // 5), random.randint(height // 5, (4 * height) // 5))
        axes = (random.randint(max(8, width // 20), max(12, width // 7)), random.randint(max(8, height // 20), max(12, height // 7)))
        angle = random.randint(0, 180)
        cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
        if random.random() > 0.5:
            pt2 = (min(width - 1, center[0] + random.randint(-axes[0], axes[0])), min(height - 1, center[1] + random.randint(-axes[1], axes[1])))
            cv2.line(mask, center, pt2, 255, thickness=random.randint(2, 6))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        return (mask > 30).astype(np.uint8)

    @staticmethod
    def _poisson_blend(base_img: np.ndarray, defect_texture: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = base_img.shape[:2]
        texture = cv2.resize(defect_texture, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_u8 = (mask * 255).astype(np.uint8)
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0 or len(ys) == 0:
            return base_img.copy()
        center = (int(np.mean(xs)), int(np.mean(ys)))
        try:
            return cv2.seamlessClone(texture, base_img, mask_u8, center, cv2.NORMAL_CLONE)
        except Exception:
            out = base_img.copy()
            out[mask > 0] = texture[mask > 0]
            return out

    @staticmethod
    def _prompt_to_style(prompt: str) -> dict[str, Any]:
        lower = str(prompt or "").lower()
        if "crack" in lower:
            return {"color": (20, 20, 20), "thickness": 2, "line": True}
        if "pitting" in lower or "pit" in lower:
            return {"color": (50, 50, 50), "thickness": 5, "line": False}
        return {"color": (35, 35, 80), "thickness": 4, "line": False}

    def _generate_inpainted_defect(self, image: np.ndarray, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        if self.cfg.device == "cuda" and not self._gpu_has_headroom():
            raise RuntimeError("Insufficient free VRAM (>4GB required) for synthetic inpainting mode")

        h, w = image.shape[:2]
        mask = self._random_blob_mask(h, w)
        out = image.copy()
        style = self._prompt_to_style(prompt)
        if style["line"]:
            ys, xs = np.where(mask > 0)
            if len(xs) > 1:
                points = np.column_stack([xs, ys])
                points = points[np.argsort(points[:, 0])]
                cv2.polylines(out, [points.reshape(-1, 1, 2)], False, style["color"], thickness=style["thickness"])
        else:
            noise = np.zeros_like(out)
            noise[:, :] = style["color"]
            blended = cv2.addWeighted(out, 0.75, noise, 0.25, 0)
            out[mask > 0] = blended[mask > 0]
        out = cv2.GaussianBlur(out, (3, 3), 0)
        return out, mask

    def synthesize_one(
        self,
        good_image_path: str | Path,
        output_image_path: str | Path,
        output_label_path: str | Path,
        *,
        mode: str = "geometric",
        defect_texture_path: str | Path | None = None,
        prompt: str = "jagged surface crack on brushed aluminum",
        class_id: int = 0,
    ) -> dict[str, Any]:
        good_image_path = Path(good_image_path)
        output_image_path = Path(output_image_path)
        output_label_path = Path(output_label_path)

        image = cv2.imread(str(good_image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read base image: {good_image_path}")

        image = cv2.resize(image, self.cfg.output_resolution, interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]

        if mode == "geometric":
            if defect_texture_path is None:
                raise ValueError("defect_texture_path is required for geometric mode")
            texture = cv2.imread(str(defect_texture_path))
            if texture is None:
                raise FileNotFoundError(f"Could not read defect texture: {defect_texture_path}")
            mask = self._random_blob_mask(h, w)
            synthesized = self._poisson_blend(image, texture, mask)
        elif mode == "inpainting":
            synthesized, mask = self._generate_inpainted_defect(image, prompt)
        else:
            raise ValueError(f"Unsupported synthetic generation mode: {mode}")

        polygons = mask_to_polygons(mask.astype(np.uint8), min_area=20)
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_image_path), synthesized)

        lines = [f"{class_id} {polygon_to_yolo_seg_line(poly, w, h)}" for poly in polygons if len(poly) >= 3]
        output_label_path.write_text("\n".join(lines), encoding="utf-8")

        return {
            "image_path": str(output_image_path),
            "label_path": str(output_label_path),
            "polygon_count": len(polygons),
            "mode": mode,
        }

    def batch_generate(
        self,
        good_images_dir: str | Path,
        output_dir: str | Path,
        *,
        mode: str = "geometric",
        defect_textures_dir: str | Path | None = None,
        prompt: str = "jagged surface crack on brushed aluminum",
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        good_images_dir = Path(good_images_dir)
        output_dir = Path(output_dir)
        img_out = output_dir / "images"
        lbl_out = output_dir / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        good_images = self._load_images(good_images_dir)
        if limit is not None:
            good_images = good_images[: int(limit)]
        textures = self._load_images(Path(defect_textures_dir)) if defect_textures_dir else []

        outputs: list[dict[str, Any]] = []
        for idx, good_img in enumerate(good_images):
            texture_path = random.choice(textures) if textures else None
            out_img = img_out / f"synthetic_{idx:05d}{good_img.suffix.lower()}"
            out_lbl = lbl_out / f"synthetic_{idx:05d}.txt"
            outputs.append(
                self.synthesize_one(
                    good_img,
                    out_img,
                    out_lbl,
                    mode=mode,
                    defect_texture_path=texture_path,
                    prompt=prompt,
                )
            )
        return outputs


def inject_synthetic_data(base_path: str | Path, synthetic_path: str | Path, ratio: float = 0.8) -> dict[str, int]:
    base_path = Path(base_path)
    synthetic_path = Path(synthetic_path)
    ratio = max(0.0, min(1.0, float(ratio)))

    base_images = sorted((base_path / "images" / "train").glob("*"))
    base_labels = {p.stem: p for p in (base_path / "labels" / "train").glob("*.txt")}
    synth_images = sorted((synthetic_path / "images").glob("*"))
    synth_labels = {p.stem: p for p in (synthetic_path / "labels").glob("*.txt")}

    max_inject = int(len(base_images) * ratio) if base_images else len(synth_images)
    selected = synth_images[:max_inject]
    injected = 0

    target_img_dir = base_path / "images" / "train"
    target_lbl_dir = base_path / "labels" / "train"
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path in selected:
        lbl_path = synth_labels.get(img_path.stem)
        if lbl_path is None:
            continue
        dst_img = target_img_dir / img_path.name
        dst_lbl = target_lbl_dir / lbl_path.name
        shutil.copyfile(img_path, dst_img)
        shutil.copyfile(lbl_path, dst_lbl)
        injected += 1

    return {
        "base_images": len(base_images),
        "synthetic_candidates": len(synth_images),
        "injected": injected,
        "ratio": ratio,
    }