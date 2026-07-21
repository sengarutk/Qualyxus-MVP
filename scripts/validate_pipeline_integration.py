#!/usr/bin/env python3
import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
import cv2
import numpy as np
import yaml as pyyaml
from src.config.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineValidator")

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

def main():
    logger.info("🚀 Launching Master End-to-End Pipeline Validation Suite...")
    cfg = Config.load()
    
    raw_dataset_path = Path(cfg.get("dataset", "raw_path") or "data/raw/active_dataset")
    compiled_yaml_path = Path(cfg.get("inference", "yolo_yaml"))
    
    # Clean and re-seed the mock raw data target location
    if raw_dataset_path.exists():
        shutil.rmtree(raw_dataset_path)
    raw_dataset_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Synthesize testing assets
        generate_synthetic_dataset(raw_dataset_path, sample_count=6)
        
        # Step 2: Execute the refactored, dynamic dataset preprocessor module
        logger.info("Executing Phase 2 Dataset Ingestion Script...")
        from src.data.prepare_dataset import auto_ingest_pipeline as run_preprocessor
        run_preprocessor(src_override=str(raw_dataset_path))
        
        # Verify that the dataset configuration contract was written successfully
        if not compiled_yaml_path.exists():
            raise FileNotFoundError(f"Verification Failure: Recipe file was not compiled at: {compiled_yaml_path}")
        logger.info("✅ Phase 2 Preprocessor validation successful.")
        
        # Step 3: Execute validation cycle passing the live modified context wrapper via isolated subprocess
        # This isolates CUDA context memory and prevents VRAM leakages inside long-running host processes
        logger.info("Executing Phase 3 Decoupled Training Pipeline Entry Point...")
        
        # Write modified config to a temporary yaml file to pass it to the subprocess
        temp_config_path = Path("temp_validation_config.yaml")
        cfg.raw["training"]["epochs"] = 1
        cfg.raw["training"]["batch"] = 2
        with open(temp_config_path, "w", encoding="utf-8") as f:
            pyyaml.safe_dump(cfg.raw, f, default_flow_style=False)
            
        try:
            sub_env = os.environ.copy()
            sub_env["CONFIG_PATH"] = str(temp_config_path)
            sub_env["PYTHONPATH"] = "."
            
            script_path = Path("src/training/train_yolo_seg.py").resolve()
            logger.info(f"Spawning isolated training process with temp config: {temp_config_path}")
            subprocess.run(
                [sys.executable, str(script_path)],
                env=sub_env,
                check=True
            )
            logger.info("✅ Phase 3 Training entry-point verification successful.")
        finally:
            if temp_config_path.exists():
                os.remove(temp_config_path)
        
        logger.info("=============================================================")
        logger.info("🎉 SUCCESS: All data pipeline abstraction phases verified perfectly!")
        logger.info("=============================================================")
        
    except Exception as e:
        logger.error(f"❌ CRITICAL PIPELINE CORRUPTION VALIDATION FAILED: {str(e)}")
        sys.exit(1)
    finally:
        # Cleanup mock testing assets to preserve workspace cleanliness
        if raw_dataset_path.exists():
            shutil.rmtree(raw_dataset_path)
        processed_test_dir = Path("data/processed/active_product_recipe")
        if processed_test_dir.exists():
            shutil.rmtree(processed_test_dir)
        yaml_test_file = Path("data/processed/active_product_recipe_yolo.yaml")
        if yaml_test_file.exists():
            yaml_test_file.unlink(missing_ok=True)
        recipe_yaml_file = Path("data/processed/active_recipe.yaml")
        if recipe_yaml_file.exists():
            recipe_yaml_file.unlink(missing_ok=True)
        for cam_i in range(4):
            tmpl_file = Path(f"data/templates/golden_part_cam_{cam_i}.png")
            if tmpl_file.exists():
                tmpl_file.unlink(missing_ok=True)
        golden_main = Path("data/templates/golden_part.png")
        if golden_main.exists():
            golden_main.unlink(missing_ok=True)

if __name__ == "__main__":
    main()
