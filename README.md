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

Also included: RTSP/USB multi-camera capture, an operations console with sign-in and approvals,
a tamper-evident audit trail, backup, alerts and an optional MQTT uplink.

**Status:** tested on simulated parts, machines and PLCs, on a laptop. It has not run on a real
camera, PLC, Jetson or production line.

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

**Accounts.** Until an account exists the console reads and changes nothing. Once one exists, everyone
signs in, and the account sets the role (operator, maintenance, QA, manager). Every change needs a reason
and goes into the audit trail before it happens; the person who approves cannot also close:

```bash
python -m src.operations.auth add <name> --role QA
```

**PLC contract.** Advisory holding registers (override with `telemetry.register_map`; set
`telemetry.mode: plc` and `telemetry.endpoint` to write to a real PLC):

| Register | Meaning |
|---|---|
| 100 | Part verdict: 0 pass, 1-3 reject by severity, 4 could not be inspected (also set on a clean stop) |
| 101 | Machine cycle counter of that part, written with 100 in one frame; 0 if no machine cycle matched |
| 102 | Heartbeat counter: stops when a camera goes silent, a view cannot inspect, or a verdict fails to arrive (held at least 10 s), so treat parts as not inspected |
| 103 | Machine-level alert (geometry, vibration) that belongs to no single part |
| 104 | Sequence, 1 to 65535: **act when this changes.** It changes with every verdict and is written last. Register 101 can repeat |
| 105 | Acknowledge, written by the PLC: copy 104 here once the verdict is taken (needed with `telemetry.require_ack`) |
| 106 | Cavity of that part, on multi-cavity stations only |
| 107 | Trigger, written by the PLC (only with `inspection.trigger`): add 1 when a part is in position. Register 101 then holds this number |

A verdict stays at least `min_verdict_hold_s` (or, with `require_ack`, until acknowledged) and is
never written later than `max_verdict_age_s`; one that could not be delivered stops the heartbeat
and is recorded in the audit trail. A reject found after the part's verdict was sent is sent
again under the same part number. On shutdown every part already seen gets its verdict first.
Simulated images cannot drive a real PLC unless `telemetry.bench_test: true`.

Each physical part gets one verdict. On a multi-camera station a part passes only when every
`required_view` has passed it (with none marked, every camera is required); other cameras can
reject a part but never pass it. A view missing after `inspection.view_timeout_s` gives flag 4.
A machine cycle whose part no camera reported also gives flag 4 (`time`-linked stations, see
`inspection.expect_part_per_cycle`).

**Operations console.** Live views of every face, a trace from a rejected part to its machine cycle and
the signals that moved, control charts, machine health, the audit trail, and a to-do page: approve or close
corrections and repairs, promote or roll back models, judge sampled parts, reset a baseline. No internet.
It listens on this machine only; the network needs accounts and HTTPS:

```bash
CONFIG_PATH=config.360.yaml python -m src.console       # beside a running main.py, http://127.0.0.1:8600
python -m src.system.tls make --host 192.168.1.20
python -m src.console --host 0.0.0.0 --tls-cert secrets/tls/cert.pem --tls-key secrets/tls/key.pem
python scripts/console_demo.py                          # a simulated shift in its own folder
```

**Health, alerts, metrics, logs.** `python -m src.healthcheck` and `/api/v1/health` fail for any silent
camera, held heartbeat, undelivered verdict or full disk. `alerts:` sends problems by webhook, e-mail or a
command (an andon light, an SMS modem). `/metrics` on the console is Prometheus text.
`QUALYXUS_LOG_FORMAT=json` gives one JSON object per line.

**Backup.** `QUALYXUS_BACKUP_PASSPHRASE=... python -m src.system.backup create --to /mnt/backup`, then
`verify` and `restore`. Safe while running; keys and accounts are included only encrypted.

**Data at rest.** Files are created for the service account only. Images of parts are not encrypted by this
software: put the data on an encrypted disk (LUKS), and the audit key on another one.

**Clock.** Cycles and parts share one clock that never runs backwards, so a time correction
cannot link a part to the wrong cycle.

**Telemetry API.** This machine only. The plant network needs `QUALYXUS_API_TOKEN` and HTTPS
(`api.tls_cert`, `api.tls_key`, or `api.behind_proxy`). Browser origins: `QUALYXUS_API_CORS_ORIGINS`.

**Model files.** Joblib models can run code when loaded, so each carries a keyed signature
and an unsigned or changed file is refused. To bring one in from another box, check where it
came from, then run `python -m src.inference.signed_model sign <file>`.

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
| Drift → retraining | Score drift files a retraining request; a human approves it; the model is trained on the pooled good parts, its threshold is set on an unbiased sample whose uncertain parts a person has labelled, views and alignment are inherited from the current model, and it is registered as shadow | `... model_registry requests / approve / retrain` |
| Labelling | Review items are exported most-uncertain first with model pre-annotations (Label Studio or COCO/CVAT); polygons come back as a YOLO-seg dataset and as review verdicts | `python -m src.inference.labeling export / import` |
| Audit trail | Every verdict with image SHA-256, model versions/digests and threshold, hash-chained so edits or deletions are detectable | `python -m src.inference.audit_log verify data/audit` |
| Multi-view parts | A part fails if any view fails it; parts missing a `required_view: true` camera are reported as incomplete | console |

DINOv3 backbones (`dinov3_vits16` / `vitb16` / `vitl16`) are gated: accept Meta's DINOv3
license on Hugging Face with your own account and export `HF_TOKEN` before using them.

Validate the whole flywheel in simulation, against a twin process with identical
disturbances and no intervention:

```bash
python scripts/simulate_closed_loop.py --vision patchcore --learn 2500 --evaluate 2500
```

Simulation results are evidence that the mechanics work, not a forecast for a real line.

The simulated operator does the requested repairs (`--repair-delay`) and applies the suggested
settings. While a component is drifting the system asks for a repair and suggests no settings,
because settings would hide the wear. After a repair, or a new mould or material:

```bash
python -m src.process.baseline reset --machine imm_1 --by NAME --reason "new check ring"
python -m src.process.baseline reset --machine imm_1 --process-change --by NAME --reason "mould 12"
```

Machine values listed in `process.baseline_reset_on` (a mould id, say) do the second one by
themselves. Several machines go under `process.machines`; each has its own baseline and model.

## On a real station

* **Is a part there?** Whatever stands out from the light on the empty station. With a patterned
  background, or a part that reaches the frame edge, learn the empty station first
  (`python -m src.camera.presence learn --source rtsp://... --out data/templates/empty_top.png`) and set
  `presence: {reference: ...}` on that camera. Or let the PLC say so: `inspection.trigger` (polled, so for
  parts at rest; a moving part needs a camera wired to the sensor).
* **One model per face:** give each camera `model_view` and a model for `<recipe>-<view>` (see `config.360.yaml`).
* **Bad images are not bad parts:** light, glare, covering, position, focus, noise and compression are compared
  with the calibration images; outside them the part is *not inspected*. Recalibrate old models to get this.
* **Keeping it running:** `process.retention_days`, `logging.retention_days`, `good_pool.max_gb`, `review.max_gb`.
  Move old audit days out with `python -m src.inference.audit_log archive --before 2027-01-01 --to /mnt/archive --key-file ...`.

## Setup

Python 3.12. `requirements.txt` is for an x86 PC with an NVIDIA GPU, pinned with hashes:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.txt
pip install -r requirements-dev.txt          # tests and checks
```

**Dependencies.** Edit `requirements.in`, then
`uv pip compile requirements.in --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes --torch-backend cu126 --emit-index-url -o requirements.txt`.

Configuration is `config.yaml`, or `CONFIG_PATH=/path/to/other.yaml`. A misspelt key stops the start.

Labelled defects (images and masks) for supervised segmentation:
`python -m src.data.prepare_dataset --src /path/to/catalog/<recipe_name>`.

## Run

```bash
python main.py                  # the runtime, and its API on 127.0.0.1:8000
python -m src.console           # the console, on 127.0.0.1:8600
```

On a plant box, without Docker (as root; systemd services, its own account, health timer):

```bash
deploy/install.sh
```

With Docker: `docker compose up -d`. On a Jetson build with `--target jetson` (written, not yet run on one).

## Tests

```bash
python -m pytest                           # unit and integration tests
python scripts/smoke_360.py                # the whole six-view station, offline, 150 s
python scripts/soak.py --minutes 120       # growth, leaks, unanswered parts
ruff check . && mypy                       # style and types
```
