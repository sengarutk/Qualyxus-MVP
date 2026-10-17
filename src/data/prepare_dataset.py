import os
import shutil
import glob
import logging
import yaml
import cv2
import numpy as np
import argparse
from src.config.config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DynamicPreprocessor")

def auto_ingest_pipeline(src_override=None, clear_cache=False):
    cfg = Config.load()
    dataset_name = cfg.get("dataset", "name", default="active_product_recipe")

    # Establish local workspace directory boundaries
    raw_base_dir = os.path.abspath(cfg.get("paths", "raw_dir", default="data/raw") or "data/raw")
    recipe_raw_dir = os.path.join(raw_base_dir, dataset_name)
    raw_images_dir = os.path.join(recipe_raw_dir, "images")
    raw_masks_dir = os.path.join(recipe_raw_dir, "masks")
    template_dir = os.path.abspath("data/templates")

    # Processed directories
    processed_base = os.path.abspath(
        os.path.join(cfg.get("paths", "processed_dir", default="data/processed") or "data/processed", dataset_name)
    )
    train_img_out = os.path.join(processed_base, "images", "train")
    val_img_out = os.path.join(processed_base, "images", "val")
    train_lbl_out = os.path.join(processed_base, "labels", "train")
    val_lbl_out = os.path.join(processed_base, "labels", "val")

    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(raw_masks_dir, exist_ok=True)
    os.makedirs(template_dir, exist_ok=True)
    os.makedirs(train_img_out, exist_ok=True)
    os.makedirs(val_img_out, exist_ok=True)
    os.makedirs(train_lbl_out, exist_ok=True)
    os.makedirs(val_lbl_out, exist_ok=True)

    # 1. Dynamic Multi-Part Subfolder Routing Bridge
    if src_override:
        part_source_dir = os.path.abspath(src_override)
    else:
        # Master catalog of perfect-part captures: --src > QUALYXUS_PART_CATALOG env >
        # dataset.part_catalog_dir config > ~/Qualyxus_Training_Data/perfect_parts
        external_base = (
            os.getenv("QUALYXUS_PART_CATALOG")
            or cfg.get("dataset", "part_catalog_dir", default=None)
            or os.path.join(os.path.expanduser("~"), "Qualyxus_Training_Data", "perfect_parts")
        )
        part_source_dir = os.path.join(external_base, dataset_name)

    if not os.path.exists(part_source_dir):
        logger.error(f"Target subfolder path not found inside master catalog repository: {part_source_dir}")
        logger.error("Please verify that your active recipe name matches your subfolder nomenclature exactly.")
        return

    # Handle forced cache synchronization purge (restricted to raw staging repositories)
    if clear_cache:
        logger.info("Forced cache synchronization purge requested. Clearing local raw image and mask directories...")
        for target_dir in [raw_images_dir, raw_masks_dir]:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)

    # Automatically migrate images and masks if the local workspace processing cache is empty or incomplete
    if len(os.listdir(raw_images_dir)) == 0 or len(os.listdir(raw_masks_dir)) == 0:
        logger.info(f"Discovered active part data catalog inside {part_source_dir}. Syncing workspace storage...")
        
        # Check if there are separate masks directories or if they are in the same folder
        inner_masks_dir = os.path.join(part_source_dir, "masks")
        sibling_masks_dir = os.path.join(os.path.dirname(part_source_dir), "masks")
        
        def _copy_mask_file(src_path, dst_dir):
            base_name = os.path.basename(src_path)
            lower_name = base_name.lower()
            if lower_name.endswith("_mask.png"):
                norm_name = base_name[:-9] + ".png"
            elif lower_name.endswith("_mask.jpg"):
                norm_name = base_name[:-9] + ".jpg"
            else:
                norm_name = base_name
            shutil.copy(src_path, os.path.join(dst_dir, norm_name))

        # Migrate raw images
        for fpath in glob.glob(os.path.join(part_source_dir, "*.png")):
            fname = os.path.basename(fpath).lower()
            if fname.endswith('_mask.png') or "/masks/" in fpath.replace("\\", "/"):
                _copy_mask_file(fpath, raw_masks_dir)
            else:
                shutil.copy(fpath, raw_images_dir)
                
        # Migrate mask directories if present
        if os.path.exists(inner_masks_dir):
            logger.info(f"Syncing masks from inner source repository masks directory: {inner_masks_dir}")
            for fpath in glob.glob(os.path.join(inner_masks_dir, "*.png")):
                _copy_mask_file(fpath, raw_masks_dir)
        elif os.path.exists(sibling_masks_dir):
            logger.info(f"Syncing masks from sibling source repository masks directory: {sibling_masks_dir}")
            for fpath in glob.glob(os.path.join(sibling_masks_dir, "*.png")):
                _copy_mask_file(fpath, raw_masks_dir)

    raw_files = glob.glob(os.path.join(raw_images_dir, "*.png"))
    if not raw_files:
        logger.error(f"Zero raw production images discovered inside workspace directory: {raw_images_dir}")
        return

    logger.info(f"Successfully loaded and staged {len(raw_files)} frames for recipe profile: [{dataset_name}].")

    # 2. Perspective-Aware & Universal Template Anchor Extraction
    universal_template_path = os.path.join(template_dir, "golden_part.png")
    if raw_files:
        shutil.copy(raw_files[0], universal_template_path)
        logger.info("Universal fallback anchor written to data/templates/golden_part.png")

    for i in range(4):
        cam_tag = f"cam_{i}"
        cam_spec_src = os.path.join(part_source_dir, f"golden_part_{cam_tag}.png")
        dest_template = os.path.join(template_dir, f"golden_part_{cam_tag}.png")
        if os.path.exists(cam_spec_src):
            shutil.copy(cam_spec_src, dest_template)
            logger.info(f"Copied perspective-aware template for camera channel {cam_tag}: {cam_spec_src}")
        else:
            logger.warning(f"Camera-specific template '{cam_spec_src}' not found. Falling back to universal master template.")
            if os.path.exists(universal_template_path):
                shutil.copy(universal_template_path, dest_template)

    # 3. Dynamic Partition Splitting and Target Output Creation
    # Always clear the split directories before partitioning to prevent split cross-contamination
    for target_dir in [train_img_out, val_img_out, train_lbl_out, val_lbl_out]:
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

    np.random.seed(42)
    np.random.shuffle(raw_files)
    split_idx = int(len(raw_files) * 0.8)

    total_instances_traced = 0

    for idx, fpath in enumerate(raw_files):
        fname = os.path.basename(fpath)
        is_train = idx < split_idx

        dest_img_folder = train_img_out if is_train else val_img_out
        dest_lbl_folder = train_lbl_out if is_train else val_lbl_out

        shutil.copy(fpath, os.path.join(dest_img_folder, fname))

        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(dest_lbl_folder, lbl_name)

        mask_path = os.path.join(raw_masks_dir, fname)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Target mask file was corrupted or unreadable: {mask_path}. Skipping file.")
                with open(lbl_path, "w") as lf:
                    pass
                continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            with open(lbl_path, "w") as lf:
                for txt_contour in contours:
                    if cv2.contourArea(txt_contour) > 25:
                        total_instances_traced += 1
                        h, w = mask.shape[:2]
                        normalized_coords = []
                        for pt in txt_contour:
                            normalized_coords.append(f"{pt[0][0] / w:.6f} {pt[0][1] / h:.6f}")
                        lf.write(f"0 {' '.join(normalized_coords)}\n")
        else:
            # No mask present – generate an empty label file to treat as background (unsupervised baseline)
            with open(lbl_path, "w") as lf:
                pass

    if total_instances_traced == 0:
        logger.warning("Zero real anomalies found. Structuring unsupervised baseline dataset profiles with empty label files per YOLOv8 specification.")

    # 4. Compile the Unified YOLOv8 Deployment Contract
    contract_data = {
        "path": processed_base,
        "train": "images/train",
        "val": "images/val",
        "test": "images/val",
        "names": {int(k): str(v) for k, v in cfg.get("dataset", "classes", default={0: "defect"}).items()}
    }

    contract_output = cfg.get("inference", "yolo_yaml")
    os.makedirs(os.path.dirname(contract_output) or ".", exist_ok=True)
    with open(contract_output, "w") as yf:
        yaml.safe_dump(contract_data, yf, default_flow_style=False)

    logger.info(f"✅ Automated plug-and-play dataset processing contract compiled at {contract_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess metrology datasets dynamically.")
    parser.add_argument("--src", type=str, default=None, help="Custom source directory for perfect part images")
    parser.add_argument("--clear-cache", action="store_true", help="Force clear local raw and processed image scratch directories before syncing master repository contents.")
    args = parser.parse_args()

    auto_ingest_pipeline(src_override=args.src, clear_cache=args.clear_cache)