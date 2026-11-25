"""Long run of the six-view station: does anything grow, leak, slow down or go unanswered?

    python scripts/soak.py --minutes 120 --out reports/soak.json

Runs in a temporary folder with simulated parts. Memory, threads, open files and disk are
sampled every 30 s; at the end every machine cycle must have exactly one verdict, every demo
defect must be rejected, the audit trail must be intact and nothing may have logged an error.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sample(proc: psutil.Process, work: str, started: float) -> dict:
    size = sum(f.stat().st_size for f in Path(work, "data").rglob("*") if f.is_file()) if Path(work, "data").exists() else 0
    return {"t": round(time.time() - started), "rss_mb": round(proc.memory_info().rss / 1e6, 1),
            "threads": proc.num_threads(), "files": proc.num_fds(), "disk_mb": round(size / 1e6, 1)}


def growth_per_hour(samples: list[dict], key: str) -> float:
    """Slope after the first third, which is warm-up."""
    tail = samples[len(samples) // 3:]
    if len(tail) < 4:
        return 0.0
    return float(np.polyfit([s["t"] / 3600 for s in tail], [s[key] for s in tail], 1)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--out", default="reports/soak.json")
    args = parser.parse_args()
    from src.inference.audit_log import verify_chain

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as work:
        env = {**os.environ, "CONFIG_PATH": str(ROOT / "config.360.yaml"), "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1",
               "TRANSFORMERS_OFFLINE": "1", "QUALYXUS_BACKBONE_DIR": str(ROOT / "models" / "backbones")}
        os.symlink(ROOT / "models", Path(work) / "models")
        log_path = Path(work) / "run.log"
        started, samples = time.time(), []
        with open(log_path, "w", encoding="utf-8") as log:
            run = subprocess.Popen([sys.executable, str(ROOT / "main.py")], cwd=work, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
            proc = psutil.Process(run.pid)
            while time.time() - started < args.minutes * 60:
                time.sleep(30)
                if run.poll() is not None:
                    problems.append(f"the runtime ended by itself with code {run.returncode}")
                    break
                samples.append(sample(proc, work, started))
                print(samples[-1], flush=True)
            if run.poll() is None:
                run.send_signal(signal.SIGINT)
                try:
                    run.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    run.kill()
                    problems.append("the runtime did not stop within 60 s")
        errors = [ln for ln in log_path.read_text(errors="replace").splitlines() if "[ERROR]" in ln or "[CRITICAL]" in ln
                  or "Traceback" in ln]
        if errors:
            problems.append(f"{len(errors)} error lines, first: {errors[0][:200]}")
        records = [json.loads(line) for f in sorted(glob.glob(f"{work}/data/audit/*.jsonl")) for line in open(f)]
        verdicts = [r for r in records if r.get("record") == "part_verdict" and not r.get("update")]
        done = [r for r in verdicts if not r.get("shutdown")]
        counts = collections.Counter(r["verdict"] for r in done)
        repeats = [c for c, n in collections.Counter(r["cycle_id"] for r in verdicts if r["cycle_id"]).items() if n > 1]
        with sqlite3.connect(f"{work}/data/events/process.db") as db:
            cycles = {r[0] for r in db.execute("SELECT cycle_id FROM process_cycles ORDER BY end_ts")}
            newest = {r[0] for r in db.execute("SELECT cycle_id FROM process_cycles ORDER BY end_ts DESC LIMIT 6")}
        judged = {r["cycle_id"] for r in verdicts}
        wrong = [r["part"] for r in done if r["verdict"] != "not_inspected" and r["part"] % 5 == 0 and r["verdict"] != "fail"]
        audit = verify_chain(f"{work}/data/audit", Path(work) / "secrets/audit.key")
        if missing := cycles - judged - newest:
            problems.append(f"{len(missing)} machine cycles got no verdict")
        if repeats:
            problems.append(f"{len(repeats)} parts were judged twice")
        if wrong:
            problems.append(f"demo defects passed: {wrong[:5]}")
        if counts["not_inspected"] > 0.01 * max(len(done), 1):
            problems.append(f"{counts['not_inspected']} of {len(done)} parts not inspected")
        if not audit["ok"]:
            problems.append(f"audit trail: {audit['error']}")
        slopes = {k: round(growth_per_hour(samples, k), 2) for k in ("rss_mb", "threads", "files", "disk_mb")}
        if slopes["rss_mb"] > 60 or slopes["threads"] > 2 or slopes["files"] > 5:
            problems.append(f"something grows: {slopes} per hour")
    good = [r for r in done if r["part"] % 5]
    report = {"minutes": round((time.time() - started) / 60, 1), "python": sys.version.split()[0], "parts": len(done),
              "verdicts": dict(counts), "cut_short_by_stop": len(verdicts) - len(done), "cycles": len(cycles),
              "rejected_without_demo_defect": sum(r["verdict"] == "fail" for r in good), "without_demo_defect": len(good),
              "audit_records": audit["records"], "growth_per_hour": slopes, "first": samples[:1], "last": samples[-1:],
              "problems": problems}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
