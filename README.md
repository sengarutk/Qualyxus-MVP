# Qualyxus — Process-Aware Visual Inspection

Qualyxus links **what a part looks like** (vision) to **how it was made** (machine
telemetry) at the level of each individual part, and uses that link to catch defects,
explain them, predict them before the part reaches the camera, and recommend the
setpoint correction that prevents them.

```
machine cycle ──► per-cycle features ──► defect-risk score ──► alerts + approval-gated prescriptions
      │                                        ▲
      └── transport ──► camera ──► per-part verdict (good-parts-only anomaly model)
                                       │  linked back to its producing cycle (alignment)
                                       └──────────► training labels ──► retrained risk model
```

Also included: RTSP/USB multi-camera capture, batched YOLO inference, golden-template
geometry checks, SQLite/CSV audit logging, an SSE API, a Streamlit dashboard and
ISO 9001 compliance reporting.

## The process ↔ quality loop

| Piece | Where |
|---|---|
| Cycle sources: simulator, Modbus TCP, OPC UA (cycle-counter segmentation) | `src/process/sources.py` |
| Per-cycle curve features + golden-cycle deviation | `src/process/features.py` |
| Image ↔ cycle alignment (`time` / `fifo` / `part_id`) + transport-delay calibration | `src/process/alignment.py` |
| Risk model: supervised (vision-labelled) + datasheet spec limits; deviation model before labels | `src/process/risk_model.py` |
| Regime-aware baselining after intentional setpoint changes | `src/process/service.py` |
| Bounded, explained, approval-gated setpoint prescriptions | `src/process/prescriptions.py` |
| KPIs: scrap, risk precision/recall, early-warning lead, false rejects/escapes (vs audits) | `src/process/kpis.py` |
| Good-parts-only PatchCore anomaly detector | `src/inference/anomaly/` |
| Per-part inspection (one pass/fail record per physical part) | `src/camera/part_inspection.py` |
| Injection-molding simulator + synthetic part renderer (development/validation) | `src/process/simulator.py`, `src/process/synthetic_parts.py` |

Configure it in the `process:`, `inspection:` and per-camera `process_link:` sections of
`config.yaml`. To connect a real machine, set `process.source` to `modbus` or `opcua` and
map the signals (examples are in `config.yaml`); calibrate the camera's transport delay
with `src.process.alignment.estimate_transport_delay`.

## Running on a plant network

**Air-gapped install.** Bundle the backbones on a connected machine, then ship `models/`:

```bash
python -m src.inference.anomaly.bundle --model models/anomaly/<recipe>.pt --verify
```

`--verify` reloads the model the way a clean disconnected box would, with an empty download
cache and the model hubs disabled. The Docker image keeps hub access off at runtime.

**Accounts.** Until an account exists the dashboard is read-only. Once one exists, everyone
must sign in, and the account sets the role:

```bash
python -m src.operations.auth add <name> --role QA
```

**PLC contract.** Advisory holding registers (override with `telemetry.register_map`; set
`telemetry.mode: plc` and `telemetry.endpoint` to write to a real PLC):

| Register | Meaning |
|---|---|
| 100 | Part verdict: 0 pass, 1-3 reject by severity, 4 could not be inspected (also set on a clean stop) |
| 101 | Machine cycle counter of that part (changes with every verdict; written with 100 in one frame) |
| 102 | Heartbeat counter: stops when a camera goes silent, a view cannot inspect, or a verdict fails to arrive (held at least 10 s), so treat parts as not inspected |
| 103 | Machine-level alert (geometry, vibration) that belongs to no single part |

Each physical part gets one verdict. On a multi-camera station a part passes only when every
`required_view` has passed it; a view missing after `inspection.view_timeout_s` gives flag 4.

**Telemetry API.** It listens on this machine only. To open it to the plant network, set
`api.host`, and set `QUALYXUS_API_TOKEN` so the streams need a bearer token. Browser origins
are listed in `QUALYXUS_API_CORS_ORIGINS`.

**Audit trail.** Records are chained with a keyed hash, and `data/audit.anchor.json` catches
deleted newest records, even across a restart. Keep `inspection.audit.key_file` away from the log on a real line, and
copy the anchor off the box regularly.

## Vision model stack

| Role | Models | License | When |
|---|---|---|---|
| Good-parts-only detection | **Dinomaly** (CVPR 2025; DINOv2-reg encoder, noisy-bottleneck linear-attention decoder), **PatchCore** on WideResNet-50 or DINOv2/DINOv3 patch tokens (AnomalyDINO recipe) | Apache-2.0 / BSD (DINOv3: Meta custom, opt-in) | From day one - needs only good parts |
| Supervised defect segmentation | **RF-DETR-Seg** Nano–Large (default), YOLO26-seg (opt-in) | Apache-2.0 (RF-DETR), AGPL-3.0 (Ultralytics) | Once labelled defects exist |
| Defect typing | DINOv2 crop embeddings + logistic regression, novelty → `unknown` | Apache-2.0 | A few operator-tagged crops per type |

The right anomaly model differs by product (measured below), so it is chosen per recipe by
`src.inference.anomaly.benchmark`. Candidates are ranked by **false-reject rate at zero
escapes** (FRR @ 100% defect recall), then FRR @ 99%, AU-PRO and image AUROC, within an
optional latency budget. Thresholds are set by **split-conformal calibration** on held-out
good parts, a distribution-free bound on the false-reject rate; with `--confidence 0.9`
the bound holds for *this* deployed model with 90% probability rather than on average
(needs ~230 held-out good images for a 1% target). Every part's verdict
feeds **model-health monitoring** (score drift and reject-rate excess, no labels needed),
and borderline parts go to a **review queue** whose labels become audits and typing data.
Dinomaly and PatchCore models export to **ONNX** (`src.inference.anomaly.export`) with a
numerical parity check (and, with `--calibration-images`, a threshold re-derived on the
exported model's own scores), served with ONNX Runtime (TensorRT/CUDA providers when installed).

In front of any detector, **part registration** (`--register`) finds the part, rotates it to
a canonical pose (resolving 180° flips against a template) and crops it, so loose fixturing
does not read as an anomaly; **tiling** (`--tiles 2x2`) scores high-resolution tiles so
small defects on large parts are not lost to resizing. Maps are warped back to camera
coordinates. A score-level **ensemble** (`ensemble_dinomaly_patchcore_s`) fuses Dinomaly and
PatchCore with robust per-model normalisation from a disjoint half of the hold-out set.

Benchmark on real data (VisA, CC BY 4.0) - results are written to `reports/anomaly_benchmark_visa.md`:

```bash
python -m src.inference.anomaly.benchmark --visa data/benchmarks/visa_raw --categories pcb1,capsules --select
```

Train the anomaly model on images of **good** parts only:

```bash
python -m src.inference.anomaly.benchmark --good-dir data/raw/<recipe>/images --select --save-best models/anomaly/<recipe>.pt
python -m src.inference.anomaly.train --images data/raw/<recipe>/images --method dinomaly --backbone dinov2reg_vitb14 --out models/anomaly/<recipe>.pt
python -m src.inference.anomaly.train --synthetic 120 --out models/anomaly/synthetic_parts.pt   # simulator scene
```

Supervised segmentation and typing, once labelled defects exist:

```bash
python -m src.inference.segmentation.train --data data/processed/active_recipe.yaml --out models/segmentation/<recipe>
python -m src.inference.defect_typing --crops data/labels/defect_crops --out models/typing/<recipe>.joblib
```

### Model operations

| Concern | What happens | Tooling |
|---|---|---|
| Versions & rollback | Each recipe/role has immutable versions, an active one and a full who/when/why history; promote and rollback are picked up by the running line within seconds | `python -m src.inference.model_registry register / list / promote / rollback` |
| Shadow mode | A candidate judges every live part next to the active model without affecting verdicts; the report compares reject rates and accuracy/escapes on human-reviewed parts | `... model_registry shadow-report --recipe <r>` |
| Recipes | Models are loaded per recipe: per camera (`recipe:`) or from the producing cycle's machine data (`inspection.recipe_from_scalar`, e.g. the mold id) | `config.yaml` → `inspection` |
| Drift → retraining | Score drift files a retraining request; a human approves it; the model is trained on the pooled good parts captured after the change and registered as shadow | `... model_registry requests / approve / retrain` |
| Labelling | Review items are exported most-uncertain first with model pre-annotations (Label Studio or COCO/CVAT); polygons come back as a YOLO-seg dataset and as review verdicts | `python -m src.inference.labeling export / import` |
| Audit trail | Every verdict with image SHA-256, model versions/digests and threshold, hash-chained so edits or deletions are detectable | `python -m src.inference.audit_log verify data/audit` |
| Multi-view parts | A part fails if any view fails it; parts missing a `required_view: true` camera are reported as incomplete | dashboard → Vision Models |

DINOv3 backbones (`dinov3_vits16` / `vitb16` / `vitl16`) are gated: accept Meta's DINOv3
license on Hugging Face with your own account and export `HF_TOKEN` before using them.

Validate the whole flywheel in simulation, against a twin process with identical
disturbances and no intervention:

```bash
python scripts/simulate_closed_loop.py --vision patchcore --learn 2500 --evaluate 2500
```

Simulation results are evidence that the mechanics work, not a forecast for a real line.

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

## Tests and validation

```bash
python -m pytest tests/        # unit + integration suite (includes a live OPC UA round trip)
```

Scenario scripts can be run from the repository root, e.g.:

```bash
python scripts/test_multicamera_isolation.py
python scripts/test_reliability.py
python scripts/test_vibration_engine.py
python scripts/test_geometry_engine.py
python scripts/malicious_audit.py
python scripts/validate_pipeline_integration.py   # isolated 1-epoch ingest + training run
```
