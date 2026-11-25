# Qualyxus MVP: full audit, 2026-11-24

Commit `0d06b82`. Nothing was changed during this audit. Stage 1 (marked fixed) was done on 2026-11-25; its tests are in `tests/test_safe_verdicts.py`.
Stage 2 (also marked fixed) was done on 2026-11-25; its tests are in `tests/test_true_numbers.py`.
Probes 06, 16, 17, 20 and 23 in `scripts/audit_probes.py` still fail: they check the old text or method, not the fixed behaviour.
A1 and A8 are planned for stage 3. A11 also needs the plant's PLC program to copy the sequence register to the acknowledge register.

**Evidence labels**
- **P**: proven by running it (a probe failed, or it was measured)
- **C**: confirmed by reading the code
- **J**: engineering judgement

**Effort:** S = under half a day, M = 1 to 2 days, L = 3 days or more. **HW** = needs hardware or real parts.

**How it was done:** all 140 source files mapped for use; ruff, bandit, vulture, radon and pip-audit on everything; full tests with coverage; the live runtime, inspection, process, PLC, logging, dashboard, console and registry code read; smaller and unused modules covered by the scanners and skimmed, not read line by line. 36 attack probes (29 by behaviour, 7 by inspecting the code), scaling measurements, and 18 imaging conditions plus 9 altered parts on the real model, all on simulated parts.

**Baseline:** 26,263 source lines; 186 tests pass; 53% coverage (32% for the main camera file); about 6,000 lines not used by the runtime, console, dashboard or healthcheck (some are offline tools that are needed).

---

## A. Critical: can give a plant a wrong or missing verdict

| # | Fault | Evidence | Where | Effort |
|---|---|---|---|---|
| A1 | **Part detection only works on a plain background.** An empty station with ordinary uneven light counts as "part present", so parts never separate: 0 of 10 detected. No hardware trigger input exists either. | P | `camera_manager.is_part_present` | M |
| ~~A2~~ fixed | **Repeated part number to the PLC.** When one part has no machine cycle, the next number repeats (1,2,3,4,**4**,5). The PLC contract says it changes every time, so a reject can go unnoticed. | P | `_publish_part_verdict` | S |
| ~~A3~~ fixed | **Cameras overwrite each other's verdict.** Without `required_view`, one camera's pass replaces another's reject for the same part. The shipped `config.yaml` has 5 cameras and none marked required. | P | `PartVerdictFuser`, `config.yaml` | S |
| ~~A4~~ fixed | **Old verdicts are sent late.** With the PLC unreachable, each write blocks 3 s; verdicts queue without limit and have no expiry, so a 60 s old verdict is still written, for a part long gone. | P | `_deliver_verdict`, `plc_adapter` | S |
| ~~A5~~ fixed | **Retraining breaks the false-reject guarantee.** It calibrates only on parts scoring under 0.8, so the new limit is too tight: 8.9% of good parts rejected in simulation of the method, while reporting 1% "guaranteed". | P | `good_pool`, `model_registry._retrain` | M |
| ~~A6~~ fixed | **Retraining ignores the number of views,** so six-view false rejects return. It also drops part alignment unless asked. | C | `model_registry._retrain` | S |
| ~~A7~~ fixed | **Unknown product is judged with the default product's model.** A missing recipe should be "not inspected". | P | `StackProvider._build` | S |
| A8 | **Two views of one part can link to two cycles.** 0.2 s apart on a 3 s cycle gave m-1 and m-2, so neither part completes. Each view should not link on its own. | P | `alignment.CycleAligner` | M |
| ~~A9~~ fixed | **Parts waiting at shutdown get no verdict.** The PLC thread stops first, then queued inspections are cancelled. | C | `_sync_stop` | S |
| ~~A10~~ fixed | **A typo in config silently removes a safety rule.** `requred_view` loads without complaint, so that view is no longer needed for a pass. Duplicate camera ids load too. | P | `config.py` | M |
| ~~A11~~ fixed | **No verdict handshake.** One register pair, no acknowledgement or sequence buffer: two verdicts within one PLC scan lose the first. Nothing is read back. | C | `plc_adapter` | M |
| ~~A12~~ fixed | **A real camera can be replaced by a fake feed** if `fallback_to_synthetic_on_failure` or an environment variable is set. Off in shipped configs, but the code should not exist in a production runtime. | C | `_capture_loop` | S |
| ~~A13~~ fixed | **A reject that arrives after the timeout is ignored.** The part is already "not inspected", never a pass, but the reject is lost from the verdict. | P | `PartVerdictFuser.add` | S |

## B. High: wrong numbers, lost data, or fails at real scale

| # | Fault | Evidence | Where | Effort |
|---|---|---|---|---|
| ~~B1~~ fixed | **Dashboard counts camera views as parts.** 10 parts with 1 reject shows as 60 parts, 1.7% scrap (true: 10%). Affects the old dashboard, not the new console. Early-warning lead times are inflated the same way on multi-view stations. | P | `process/kpis.py` | S |
| ~~B2~~ fixed | Parts that could not be inspected count as good in the scrap rate. | P | `kpis.py` | S |
| ~~B3~~ fixed | A part with an uninspected view trains the risk model as "good". | P | `store.labelled_dataset` | S |
| ~~B4~~ fixed | Each six-view part counts as 6 labels: retrains every 42 parts, not 250. | P | `service.record_inspection` | S |
| ~~B5~~ fixed | Retraining reloads all history every time: about 147 s of preparation after 30 days, repeated every 13 minutes. | P | `service.retrain` | M |
| ~~B6~~ fixed | **Lost machine cycles leave no record.** A counter jump of 5 gives 1 cycle. | P | `CycleAssembler.feed` | S |
| ~~B7~~ fixed | **A network outage is merged into one cycle:** a 59 s gap became a 61 s "cycle" that feeds health and training. | P | `CycleAssembler` | S |
| ~~B8~~ fixed | Health baseline is the first 100 cycles ever, for life. Maintenance, a new mould or material never resets it. | C | `machine_health`, `console/data` | M |
| ~~B9~~ fixed | Health mixes all machines, and the runtime supports one machine source only. | C | `store.cycle_features`, `main.py` | M |
| B10 | **Lighting faults are treated as bad parts.** Uneven light, shadow, reflection, JPEG compression, textured belt and half-in-frame each rejected 20 of 20 good parts. Half-covered, glare, darker and brighter frames pass the image check. No exposure or framing check. | P | `image_quality.py` | M |
| ~~B11~~ fixed | **Model-blind drift raises no alarm.** Scores falling to a tenth of normal pass silently; only upward drift alerts. | P | `monitoring.py` | S |
| ~~B12~~ fixed | **Second rollback returns to the bad model** (v3, then v2, then v3 again). | P | `model_registry.rollback` | S |
| ~~B13~~ fixed | Model file digest is recorded but never checked at load (vision models; joblib ones are signed). | C | `inspection_stack` | S |
| B14 | **4.6 GB RAM per 12 MP camera** for the frame buffer, 28 GB for six. | P | `_POOL_SLOTS` | S |
| B15 | **5 fps limit at 12 MP** from presence, sharpness and copying on full frames. | P | `part_inspection`, `camera_manager` | M |
| B16 | Storage at 12 MP: up to 69 GB per recipe (good pool) and 115 GB per camera (review). Neither obeys the disk monitor. | P | `good_pool`, `monitoring` | S |
| B17 | No retention for cycles, inspections, events or audit: about 82 MB a day in `process.db` per line. | P | `store.py` | M |
| ~~B18~~ fixed | Audit check reads the whole log every 10 s: 11 s long after 30 days, 129 s after a year. Needs checkpoints. | P | `console`, `audit_log` | M |
| ~~B19~~ fixed | **Event writer dies on one database error** ("database is locked"), and tracking events in that batch are lost. It restarts only on the next event. | P | `event_logger` | S |
| ~~B20~~ fixed | Events already taken from the queue are lost at shutdown. | P | `event_logger` | S |
| ~~B21~~ fixed | **Random telemetry is written into event records** in default mode: made-up temperature, vibration, lot and shift, then hashed as evidence. | P | `plc_adapter.read_context` | S |
| ~~B22~~ fixed | **Language-model errors are saved as the defect's root cause.** It is on by default and sends every defect to a local server that may not exist. | P | `vlm_gateway` | S |
| ~~B23~~ fixed | One 95 °C reading latches EMERGENCY: all cameras stop until restart. No confirmation, no recovery. | P | `resource_monitor` | S |
| ~~B24~~ fixed | Corrections advise raising hold pressure when the alarm says the check ring is worn, which hides the wear. Old suggestions stay open and contradict newer ones. | C | `prescriptions.py` | M |
| B25 | Modbus reads one signal per request (10 round trips per sample); 20 Hz needs under 5 ms each. Use block reads. One failed register drops the whole sample and reconnects. | C | `ModbusCycleSource` | M |
| B26 | **Docker image cannot run on a Jetson:** x86 CUDA wheels on an x86-style base. The image has never been built to completion. | C | `Dockerfile`, `requirements.txt` | L |
| B27 | Python 3.10 reached end of life in October 2026. 48 advisories in pillow (42), tornado (4) and filelock (2); this scan could not check the CUDA builds of torch, where about 20 were found earlier. | P | `requirements.txt` | L |
| B28 | OPC UA has no security policy, certificates or login wiring; bad-quality values are not handled. | C | `sources.py` | M |
| B29 | RTSP open and read have no timeout, so a hung stream can hold a thread and the camera. | C | `_open_camera` | S |
| B30 | All six views are inspected one after another on one thread, not batched: about 170 ms per part before audit writes. | C | `inspection_executor` | M |

## C. Product and claim gaps

| # | Gap | Evidence | Effort |
|---|---|---|---|
| C1 | **The "six faces" in the demo are the same face** (0.90 to 0.94 alike). One model is shared by all views; real faces differ and each needs its own. | P | M |
| C2 | Full VisA is 4 of 12 categories. MVTec AD, AD 2 and LOCO not run. | C | L |
| C3 | Catch rate at the real limit is 35 to 90%, the main AI weakness. | C | L |
| C4 | Early warning is weak: 16 of 60 episodes, median lead zero. | C | L |
| C5 | No industrial camera support: GigE Vision, USB3 Vision, GenICam, hardware trigger, strobe, exposure control. | C | L, HW |
| C6 | No FOCAS, MTConnect, Siemens S7, EtherNet/IP or PROFINET; no accelerometer; 20 Hz only. | C | L, HW |
| C7 | Never run on a real PLC, Jetson, camera or line. No run longer than minutes. | C | HW |
| C8 | Several hundred good parts needed, plus 600 or more to calibrate six views. Competitors claim 20 to 30. | C | L |
| C9 | No operator screens for changeover, training, calibration, review or approving a model; all command line. `set_recipe` has no caller. | C | L |
| C10 | No part-code reading (barcode, DataMatrix), so part identity is inferred from time. | C | M |
| C11 | No measurement or gauging, and no logical checks (count, presence, position). | J | L |
| C12 | No gauge study (MSA), limit samples or daily check-part routine, which IATF 16949 plants expect. | J | M |
| C13 | Pixel sizes are not calibrated to mm, so "smallest defect" cannot be stated. | C | M, HW |
| C14 | Defect naming is unused: everything shows as "Anomaly". | C | M |
| C15 | Reviewed parts are not fed back automatically; a command must be run. | C | M |
| C16 | No SPC export, shift report, MES or ERP link. Compliance reports exist but nothing calls them and they are untested. | C | M |
| C17 | No Jetson or TensorRT build, and no benchmark. | C | L, HW |
| C18 | Mechanical design not reviewed: no CAD in the repo. | C | HW |

## D. Security

| # | Fault | Evidence | Effort |
|---|---|---|---|
| D1 | No HTTPS on the dashboard, console or API. | C | M |
| D2 | Console token accepted in the URL (history, logs); cookie not marked secure; one shared token, no per-person login. | C | S |
| D3 | A removed or demoted user keeps their role for up to 8 hours. | C | S |
| D4 | Anyone can lock an account by failing 5 times; lockout resets on restart. | C | S |
| D5 | Model promote, rollback and approve take any name as `--by`, and approvals are not in the audit chain. | C | M |
| D6 | The same person can approve and close an action; no reason needed. | C | S |
| D7 | MQTT is on by default, unencrypted on port 1883, with no login; the buffer is lost on restart. | C | S |
| D8 | Audit key defaults to the same disk as the log. | C | S |
| D9 | Model downloads are not pinned to a revision; no hashes in requirements. | P | S |
| D10 | No backup or restore for databases, keys, models or audit. | C | M |
| D11 | No protection for data at rest; images may be sensitive for defence customers. | J | M |

## E. Reliability and operations

| # | Fault | Evidence | Effort |
|---|---|---|---|
| E1 | No soak test ever finished. | C | S |
| E2 | No systemd service or install path except Docker. | C | S |
| E3 | Healthcheck passes if any one camera works; Docker checks only the API port. | C | S |
| E4 | Two capture threads can briefly run for one camera after a restart, and the old part visit carries over. | C | S |
| E5 | The camera "sync" does not synchronise anything; it can only delay frames 50 ms. Remove it. | C | S |
| E6 | Two MQTT stacks and two event paths. One event stream endpoint never sends data. | C | M |
| E7 | CSV files pile up per camera session: 100+ in `data/events` already. | P | S |
| E8 | Full-disk image writes fail silently (`cv2.imwrite` result ignored) in the review queue and good pool. | C | S |
| E9 | Review queue and good pool list their folders on every add, slowing inspection as they grow. | C | S |
| E10 | Shared singletons (`FactoryManager`, `EventLogger`, `ConfigVault`) make tests leak state. | C | L |
| E11 | Time handling: mixed naive UTC, local and monotonic; events and telemetry still use the system clock. | C | M |
| E12 | No structured logs, metrics endpoint or alert delivery (email, SMS, andon). | C | M |
| E13 | Simulator cycle ids repeat after a restart, mixing old and new data. | C | S |
| E14 | WSL crashes under heavy GPU training; the laptop clock steps back 0.65 s every 32 s (handled). | P | Environment |

## F. Code quality

| # | Fault | Evidence | Effort |
|---|---|---|---|
| F1 | 53% coverage; 32% on `camera_manager`, 17% on `event_logger`, 14% on `disk_monitor` and `mqtt_bridge`, 0% on `healthcheck` and `bundle`. | P | L |
| F2 | `camera_manager.py` is over 1,800 lines; `_inspect_part` has complexity 61, `_render_operations` 82. | P | L |
| F3 | 41 places ignore errors silently. | P | M |
| F4 | Unused code: old YOLO tools, RCA and compliance generators, `infer_live`, `tracker`, `action_verifier`, `outcome_evaluator`, federated learning, Triton kernels. Delete or finish. | P | M |
| F5 | 11 of 18 scripts are unmaintained stress scripts, outside the test run. | J | S |
| F6 | Root clutter: `deep_check.py`, `find_imports.py`, `strip_bom.py`, `actual_installed.txt`, empty `qualyxus.db` and `codebase_analysis.txt`. | C | S |
| F7 | `docs/` describes the old YOLOv8 product and 10-line scaling that does not exist. Misleading if shared. | C | S |
| F8 | No LICENSE file. | P | S |
| F9 | Unused imports (19), unused variables (5), zip without strict (15), naive dates (9). | P | S |
| F10 | No type checking; GitHub CI result unknown; tests need downloaded models. | C | M |
| F11 | Misleading names and claims in comments ("Guarantees 100% transaction compliance", "opcua_compatible: True" from a placeholder). | C | S |
| F12 | Two dashboards overlap; the old one has the wrong KPIs (B1). Move approvals into the console and retire it. | J | L |

## Found while fixing stage 2

- **Setting corrections alone can raise scrap.** With B24 fixed for single settings only and repair requests ignored, simulated scrap rose on 2 of 3 seeds (12.4% to 19.8%, 5.2% to 15.4%). Now no correction is suggested while any component is drifting; with requests ignored, scrap then equals the untouched machine.
- **The earlier simulated scrap reduction (79% to 93% on the same 3 seeds) came mostly from settings that hid wear.** With repairs done 50 cycles after a request: 68%, 75% and 49%. Repairs do most of it. Simulation only, ideal vision (`--vision oracle`); rerun: `python scripts/simulate_closed_loop.py --vision oracle --seed 11`.
- A baseline assumes the first 100 cycles after a reset are healthy. After a repair someone must reset it (`python -m src.process.baseline`), or the repair request comes back.
- A Modbus or OPC UA source that stays down produces no cycle, so its lost cycles are recorded only when it returns.

## What held up

- Across 18 imaging conditions the model passed **no defective part** (0 of 360): it failed by rejecting good parts.
- Rotation to 90° and 180°, 60 px shift and ±20% brightness: no false rejects. 30° rotation and 30% darker: a few parts flagged as camera faults.
- 9 altered parts (missing hole, missing rib, 1 px scratch, wrong part): 20 of 20 rejected each. These are easy, clean simulated images, so expect much less on real parts.
- Audit tamper detection, heartbeat logic, model signing and offline running all passed their tests.

## Recommended order

1. **Stage 1: safe verdicts** (A2 to A7, A9 to A13): about a week.
2. **Stage 2: true numbers and data** (B1 to B9, B11 to B13, B18 to B24): about a week and a half.
3. **Stage 3: ready for a real camera** (A1, A8, B10, B14 to B17, B25, B28 to B30, C1): about two weeks.
4. **Stage 4: platform** (B26, B27, D, E, F): about two weeks.
5. **Stage 5: AI strength** (C2 to C4, C8, C14, C15): ongoing, needs a stable GPU.
6. **With funding or hardware** (C5 to C7, C13, C17, C18).

No audit finds everything. These are the faults found with the methods above; real hardware will reveal more.
