#!/usr/bin/env python3
"""End-to-end validation of the dataset ingest -> training entry point.

The run is fully isolated from the operator's workspace:
- synthetic frames/masks are generated in a temporary catalog directory
- a dedicated throw-away recipe name is used for data/raw, data/processed and
  models/checkpoints, so the configured production dataset is never touched
- golden templates in data/templates are backed up and restored afterwards
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os
import shutil
import logging
import subprocess
import tempfile
import cv2
import numpy as np
import yaml as pyyaml
from src.config.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineValidator")

VALIDATION_RECIPE = "__pipeline_validation__"


def generate_synthetic_dataset(raw_dir: Path, sample_count: int = 5):
    """Generates synthetic high-resolution frame pairs to simulate the live data stream."""
    masks_dir = raw_dir / "masks"
    raw_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Synthesizing {sample_count} mockup factory frame-mask asset pairs...")
    for i in range(sample_count):
        # Generate a mock 640x480 visual frame asset
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = raw_dir / f"factory_line_sample_{i}.png"
        cv2.imwrite(str(img_path), img)

        # Generate a corresponding binary anomaly mask containing a structural defect blob
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.rectangle(mask, (100 + i*20, 150), (250 + i*20, 300), 255, -1)
        mask_path = masks_dir / f"factory_line_sample_{i}.png"
        cv2.imwrite(str(mask_path), mask)


def _backup_templates(template_dir: Path, backup_dir: Path) -> list[Path]:
    saved = []
    if template_dir.is_dir():
        for tmpl in template_dir.glob("golden_part*.png"):
            shutil.copy2(tmpl, backup_dir / tmpl.name)
            saved.append(tmpl)
    return saved


def _restore_templates(template_dir: Path, backup_dir: Path) -> None:
    # Remove templates written by the validation run, then put the originals back.
    if template_dir.is_dir():
        for tmpl in template_dir.glob("golden_part*.png"):
            tmpl.unlink(missing_ok=True)
    for saved in backup_dir.glob("golden_part*.png"):
        template_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(saved, template_dir / saved.name)


def main():
    logger.info("🚀 Launching Master End-to-End Pipeline Validation Suite...")
    base_cfg = Config.load()

    paths_cfg = base_cfg.raw.get("paths", {}) or {}
    raw_dir = Path(paths_cfg.get("raw_dir") or "data/raw")
    processed_dir = Path(paths_cfg.get("processed_dir") or "data/processed")
    models_dir = Path(paths_cfg.get("models_dir") or "models/checkpoints")
    template_dir = Path("data/templates")

    scratch = Path(tempfile.mkdtemp(prefix="qualyxus_pipeline_validation_"))
    catalog_dir = scratch / "catalog" / VALIDATION_RECIPE
    template_backup = scratch / "template_backup"
    template_backup.mkdir(parents=True)
    temp_config_path = scratch / "validation_config.yaml"
    compiled_yaml_path = processed_dir / f"{VALIDATION_RECIPE}_yolo.yaml"

    # Isolated configuration: own recipe name, own compiled YAML, 1-epoch training.
    raw = dict(base_cfg.raw)
    raw["dataset"] = {**(raw.get("dataset", {}) or {}), "name": VALIDATION_RECIPE}
    raw["inference"] = {**(raw.get("inference", {}) or {}), "yolo_yaml": str(compiled_yaml_path)}
    raw["training"] = {**(raw.get("training", {}) or {}), "epochs": 1, "batch": 2}
    with open(temp_config_path, "w", encoding="utf-8") as f:
        pyyaml.safe_dump(raw, f, default_flow_style=False)

    previous_config_env = os.environ.get("CONFIG_PATH")
    _backup_templates(template_dir, template_backup)
    exit_code = 0
    try:
        os.environ["CONFIG_PATH"] = str(temp_config_path)

        # Step 1: Synthesize testing assets
        generate_synthetic_dataset(catalog_dir, sample_count=6)

        # Step 2: Execute the dynamic dataset preprocessor module
        logger.info("Executing Phase 2 Dataset Ingestion Script...")
        from src.data.prepare_dataset import auto_ingest_pipeline as run_preprocessor
        run_preprocessor(src_override=str(catalog_dir), clear_cache=True)

        # Verify that the dataset configuration contract was written successfully
        if not compiled_yaml_path.exists():
            raise FileNotFoundError(f"Verification Failure: Recipe file was not compiled at: {compiled_yaml_path}")
        logger.info("✅ Phase 2 Preprocessor validation successful.")

        # Step 3: Run training in an isolated subprocess.
        # This isolates CUDA context memory and prevents VRAM leakages inside long-running host processes
        logger.info("Executing Phase 3 Decoupled Training Pipeline Entry Point...")
        sub_env = os.environ.copy()
        sub_env["CONFIG_PATH"] = str(temp_config_path)
        sub_env["PYTHONPATH"] = str(ROOT_DIR)
        subprocess.run(
            [sys.executable, "-m", "src.training.train_yolo_seg"],
            env=sub_env,
            cwd=str(ROOT_DIR),
            check=True,
        )
        logger.info("✅ Phase 3 Training entry-point verification successful.")

        logger.info("=============================================================")
        logger.info("🎉 SUCCESS: All data pipeline abstraction phases verified perfectly!")
        logger.info("=============================================================")

    except Exception as e:
        logger.error(f"❌ CRITICAL PIPELINE CORRUPTION VALIDATION FAILED: {str(e)}")
        exit_code = 1
    finally:
        if previous_config_env is None:
            os.environ.pop("CONFIG_PATH", None)
        else:
            os.environ["CONFIG_PATH"] = previous_config_env

        # Cleanup only the artefacts that belong to the validation recipe.
        for path in (
            raw_dir / VALIDATION_RECIPE,
            processed_dir / VALIDATION_RECIPE,
            models_dir / VALIDATION_RECIPE,
        ):
            if path.exists():
                shutil.rmtree(path, ignore_errors=True)
        compiled_yaml_path.unlink(missing_ok=True)
        _restore_templates(template_dir, template_backup)
        shutil.rmtree(scratch, ignore_errors=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
