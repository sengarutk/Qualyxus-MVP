# Qualyxus — Defect Detection + Compliance Intelligence

Edge runtime for multi-camera visual inspection: RTSP/USB capture, batched YOLO
segmentation, defect tracking, geometry (golden-template) and vibration analysis,
VLM root-cause narratives, SQLite/CSV audit logging, a Streamlit operations dashboard
and ISO 9001 compliance reporting.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # pulls CUDA 12.1 torch wheels from the PyTorch index
```

Runtime configuration lives in `config.yaml` (repository root). It is loaded
automatically; set `CONFIG_PATH=/path/to/other.yaml` to use a different file.
`src/config/default.yaml` is only used when no `config.yaml` exists.

## Data → model

Prepare a dataset (pick one):

```bash
# Perfect-part catalog (images + masks/) -> YOLO-seg recipe, golden templates
python -m src.data.prepare_dataset --src /path/to/catalog/<recipe_name>

# Legacy Severstal steel dataset (train.csv + train_images/ under data/raw)
bash scripts/download_severstal.sh   # or: python -m src.dataset.severstal_prepare
```

Both write the compiled dataset YAML to `inference.yolo_yaml` from the config, which is
exactly where training reads it.

Train (values default to the `training:` section of the config):

```bash
python -m src.training.train_yolo_seg --model yolov8s-seg.pt --epochs 30 --imgsz 640
```

Weights are written to `models/checkpoints/<dataset.name>/weights/best.pt`; point
`inference.model_path` at that file for live inference.

## Run

```bash
python main.py                       # edge runtime + Pulse API on :8000 (/api/v1/health)
streamlit run src/dashboard/app.py   # operations dashboard on :8501
```

Offline inference over saved frames or a video:

```bash
python -m src.inference.infer_images --input_dir data/live/raw --save_dir data/live/defects
python -m src.inference.infer_video --video_path line.mp4 --source line_1_cam_1
```

## Docker

```bash
./deploy.sh        # docker compose build + up (inference runtime + dashboard)
```

## Validation scripts

All scripts can be run from the repository root, e.g.:

```bash
python scripts/test_multicamera_isolation.py
python scripts/test_reliability.py
python scripts/test_vibration_engine.py
python scripts/test_geometry_engine.py
python scripts/malicious_audit.py
python scripts/validate_pipeline_integration.py   # isolated 1-epoch ingest + training run
```
