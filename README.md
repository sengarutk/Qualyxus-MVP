# QualForge AI - Defect Detection + Compliance Intelligence MVP

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
Step 1: Prepare dataset (Severstal -> YOLOv8 seg)
python -m src.dataset.severstal_prepare
Step 2: Train YOLOv8s-seg
python -m src.training.train_yolo_seg --model yolov8s-seg.pt --epochs 30 --imgsz 640
Step 3: Inference on images
python -m src.inference.infer_images --input_dir data/samples --save_dir data/events
Step 4: Run dashboard
streamlit run src/dashboard/app.py

---

# ✅ 4) Config

## `src/config/default.yaml`
```yaml
paths:
  data_dir: "data"
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  events_dir: "data/events"
  reports_dir: "data/reports"
  models_dir: "models/checkpoints"
  exported_dir: "models/exported"

severstal:
  train_csv: "data/raw/train.csv"
  train_images_dir: "data/raw/train_images"

dataset:
  yolo:
    images_train: "data/processed/images/train"
    images_val: "data/processed/images/val"
    labels_train: "data/processed/labels/train"
    labels_val: "data/processed/labels/val"
    yaml_path: "data/processed/severstal_yolo.yaml"
    val_split: 0.15
    seed: 42

training:
  model: "yolov8s-seg.pt"
  epochs: 30
  imgsz: 640
  batch: 8
  device: 0

inference:
  conf: 0.35
  iou: 0.5
  save_overlay: true
  save_crops: true

logging:
  sqlite_path: "data/events/events.db"
  csv_path: "data/events/events.csv"