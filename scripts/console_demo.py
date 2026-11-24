"""Fill a separate demo folder with a simulated shift, then run the live station on from it.

Every part is rendered by the simulator and judged by the real inspection model, and every
verdict goes through the real fusion, linking, audit and process code. The only thing staged
is the machine: from cycle 181 its check ring wears faster than normal, and the process
physics decides which parts that ruins. Nothing is written to the project's own data/.

    python scripts/console_demo.py                 # then open http://127.0.0.1:8600
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HISTORY, HEALTHY = 290, 180
# Both seeds are chosen so the check ring is the only fault for the first 160 live parts.
SEED, LIVE_SEED, TIME_SCALE, TRANSPORT_S, MOULD_OPEN_S = 11, 21, 3.0, 4.0, 1.5
START = {"check_ring_wear": 0.02, "lot_moisture": 0.03}
WEAR = {"check_ring_wear": 0.0009, "cooling_fouling": 0.012}


def write_config(work: Path) -> Path:
    cfg = yaml.safe_load((ROOT / "config.360.yaml").read_text(encoding="utf-8"))   # extends is one level deep
    cfg = {k: v for k, v in cfg.items() if not k.startswith("x-")}
    cfg["extends"] = str(ROOT / "config.yaml")
    cfg["process"] = {"models_dir": "process_models", "simulator": {"seed": LIVE_SEED, "time_scale": TIME_SCALE}}
    cfg["inspection"] = {**cfg["inspection"], "view_store": {"keep_parts": 400}}
    cfg["cameras"] = [{**camera, "inject_defect_every": 0} for camera in cfg["cameras"]]
    path = work / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def backfill(cfg: dict, work: Path) -> dict:
    from src.camera.part_inspection import PartVerdictFuser, part_pose
    from src.camera.view_store import ViewStore
    from src.inference.audit_log import InspectionAuditLog, image_sha256
    from src.inference.inspection_stack import StackProvider
    from src.process.service import ProcessIntelligenceService
    from src.process.simulator import InjectionMoldingSimulator
    from src.process.synthetic_parts import render_part

    cameras = [c for c in cfg["cameras"] if c.get("enabled", True)]
    inspection = cfg["inspection"]
    stack, _ = StackProvider(inspection).get()
    if stack.empty:
        sys.exit("The inspection model did not load; see config.360.yaml for how to build it.")
    service = ProcessIntelligenceService(cfg["process"], cameras_cfg=cameras)
    audit = InspectionAuditLog(inspection["audit"]["dir"], inspection["audit"].get("key_file"))
    views = ViewStore(keep=400)
    fuser = PartVerdictFuser([c["id"] for c in cameras])
    sim = InjectionMoldingSimulator(seed=SEED, line_id="line_A")
    sim.load_scenario(START)
    period = 16.9 / TIME_SCALE + MOULD_OPEN_S / TIME_SCALE
    end = time.time() - HISTORY * period
    false_rejects = escapes = 0
    for i in range(HISTORY):
        if i == HEALTHY:
            sim.drift_per_cycle = dict(WEAR)
        cycle, outcome = sim.next_cycle(start_ts=0.0)
        duration = cycle.duration_s / TIME_SCALE
        cycle.start_ts, cycle.end_ts = end, end + duration
        end = cycle.end_ts + MOULD_OPEN_S / TIME_SCALE
        service.on_cycle(cycle)
        seen = cycle.end_ts + TRANSPORT_S + 0.3
        part_failed = False
        for index, camera in enumerate(cameras):
            defect = outcome.defect_type if cycle.sequence % len(cameras) == index else None
            frame = render_part(camera["width"], camera["height"], defect, cycle.sequence + index * 7919)
            judged = stack.judge(frame)
            verdict = "fail" if judged.failed else "pass"
            part_failed |= judged.failed
            heat = (stack.anomaly.heatmap_overlay(frame, judged.anomaly, stack.anomaly.threshold)
                    if judged.failed else None)
            views.save(cycle.cycle_id, camera["id"], frame, heat)
            audit.append({"ts": seen, "camera_id": camera["id"], "recipe": "default", "cycle_id": cycle.cycle_id,
                          "verdict": verdict, "defect_type": judged.defect_type,
                          "normalized_score": judged.normalized_score, "threshold": stack.anomaly.threshold,
                          "models": stack.versions, "image_sha256": image_sha256(frame), "backfilled": True})
            service.record_inspection(camera["id"], seen, verdict, defect_type=judged.defect_type,
                                      anomaly_score=judged.normalized_score, position=part_pose(frame))
            fused = fuser.add(cycle.cycle_id, camera["id"], 2 if judged.failed else 0, seen)
            if fused is not None:
                audit.append({"ts": seen, "record": "part_verdict", "part": cycle.sequence & 0xFFFF,
                              "cycle_id": cycle.cycle_id, "verdict": fused["verdict"], "flag": fused["flag"],
                              "views": fused["views"], "missing_views": fused["missing"], "backfilled": True})
        false_rejects += part_failed and outcome.defect_type is None
        escapes += (not part_failed) and outcome.defect_type is not None
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{HISTORY} parts inspected", flush=True)
    print(f"  against the simulator's truth: {false_rejects} false rejects, {escapes} missed defects")
    service.stop()
    return {"state": {"check_ring_wear": sim.check_ring_wear, "cooling_fouling": sim.cooling_fouling,
                      "heater_offset": sim.heater_offset, "lot_moisture": sim.lot_moisture},
            "start_sequence": sim.sequence}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--work", default=str(Path.home() / ".qualyxus" / "console_demo"))
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()
    work = Path(args.work).resolve() / time.strftime("run-%Y%m%d-%H%M%S")
    work.mkdir(parents=True)
    (work / "models").symlink_to(ROOT / "models")
    os.chdir(work)
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": str(ROOT),
                       "QUALYXUS_BACKBONE_DIR": str(ROOT / "models" / "backbones")})
    config = write_config(work)
    os.environ["CONFIG_PATH"] = str(config)

    from src.config.config import Config

    print(f"Demo folder: {work}\nInspecting a simulated shift of {HISTORY} parts with the real model...")
    scenario = backfill(Config.load().raw, work)
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    cfg["process"]["simulator"]["scenario"] = scenario
    config.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    log = open(work / "run.log", "w", encoding="utf-8")
    runtime = subprocess.Popen([sys.executable, str(ROOT / "main.py")], stdout=log, stderr=subprocess.STDOUT)
    console = subprocess.Popen([sys.executable, "-m", "src.console", "--port", str(args.port)])
    print("Live. Press Ctrl+C to stop.", flush=True)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        while runtime.poll() is None and console.poll() is None:
            time.sleep(1)
        print("Stopped unexpectedly; see run.log in the demo folder.")
    except KeyboardInterrupt:
        pass
    for process in (runtime, console):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for process in (runtime, console):
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
