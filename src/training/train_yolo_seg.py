#!/usr/bin/env python3
import os
import logging
from pathlib import Path
from ultralytics import YOLO
from src.config.config import Config, env_override_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainingOrchestrator")

def main(cfg: Config = None):
    logger.info("Initializing Decoupled YOLO Segmentation Training Pipeline...")
    if cfg is None:
        cfg = Config.load()
    # CRITICAL: Apply environment variable overrides universally
    cfg = env_override_config(cfg)
    
    dataset_name = cfg.get("dataset", "name") or "active_product_recipe"
    yolo_yaml_path = cfg.get("inference", "yolo_yaml")
    
    logger.info(f"Targeting active product recipe: {dataset_name}")
    logger.info(f"Resolving dynamic dataset recipe path: {yolo_yaml_path}")
    
    if not os.path.exists(yolo_yaml_path):
        raise FileNotFoundError(f"Compiled YOLO dataset configuration file missing: {yolo_yaml_path}")
        
    base_model_name = cfg.get("training", "model") or "yolov8s-seg.pt"
    epochs = int(cfg.get("training", "epochs") if cfg.get("training", "epochs") is not None else 30)
    imgsz = int(cfg.get("training", "imgsz") if cfg.get("training", "imgsz") is not None else 640)
    batch_size = int(cfg.get("training", "batch") if cfg.get("training", "batch") is not None else 8)
    device_target = cfg.get("training", "device") if cfg.get("training", "device") is not None else 0

    logger.info(f"Loading base neural weights model architecture: {base_model_name}")
    model = YOLO(base_model_name)
    
    models_dir = cfg.get("paths", "models_dir") or "models/checkpoints"
    project_save_dir = Path(models_dir) / dataset_name
    project_save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Launching accelerated model training optimization loop ({epochs} epochs)...")
    results = model.train(
        data=str(Path(yolo_yaml_path).resolve()),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device_target,
        project=str(project_save_dir.parent.resolve()),
        name=project_save_dir.name,
        exist_ok=True,
        workers=2
    )
    logger.info(f"✅ Training optimized successfully. Assets saved to: {project_save_dir}")

if __name__ == "__main__":
    main()
