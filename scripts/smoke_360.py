"""End-to-end smoke test: run the six-view station as a clean offline box, then check it.

Runs in a temporary folder with empty download caches, so it never touches data/ and proves
the system works with bundled models only. Exits non-zero if any check fails.

    python scripts/smoke_360.py [--seconds 150]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_CHECK = """
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(r"{app}", default_timeout=90).run()
assert not at.exception, [e.message[:200] for e in at.exception]
assert not at.error, [e.value[:200] for e in at.error]
labels = {{m.label: m.value for m in at.metric}}
assert labels.get("Audit trail") == "✅ intact", labels.get("Audit trail")
assert any("/100" in str(m.value) for m in at.metric), "no machine-health scores"
assert any(m.value == "**Part-position drift**" for m in at.markdown), "no position section"
print("parts on dashboard:", labels.get("Parts judged (8h)"))
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seconds", type=int, default=150, help="Machine health needs 20 cycles, about 130 s")
    args = parser.parse_args()

    problems: list[str] = []
    with tempfile.TemporaryDirectory() as work:
        env = {**os.environ, "CONFIG_PATH": str(ROOT / "config.360.yaml"), "PYTHONPATH": str(ROOT),
               "HF_HOME": str(Path(work) / "hf"), "TORCH_HOME": str(Path(work) / "torch"),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "QUALYXUS_BACKBONE_DIR": str(ROOT / "models" / "backbones")}
        os.symlink(ROOT / "models", Path(work) / "models")
        log_path = Path(work) / "run.log"
        with open(log_path, "w", encoding="utf-8") as log:
            run = subprocess.Popen([sys.executable, str(ROOT / "main.py")], cwd=work, env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
            try:
                run.wait(timeout=args.seconds)
                problems.append(f"runtime exited early with code {run.returncode}")
            except subprocess.TimeoutExpired:
                run.send_signal(signal.SIGINT)
                try:
                    run.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    run.kill()
                    problems.append("runtime did not shut down within 30 s")

        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        errors = [ln for ln in log_text.splitlines() if "[ERROR]" in ln or "Traceback" in ln]
        if errors:
            problems.append(f"{len(errors)} error lines, first: {errors[0][:160]}")

        records = [json.loads(line) for f in glob.glob(f"{work}/data/audit/*.jsonl")
                   for line in open(f, encoding="utf-8")]
        verdicts = [r for r in records if r.get("record") == "part_verdict"]
        counts = collections.Counter(r["verdict"] for r in verdicts)
        repeats = [p for p, n in collections.Counter(r["part"] for r in verdicts).items() if n > 1]
        print(f"parts: {len(verdicts)}  verdicts: {dict(counts)}")
        if not verdicts:
            problems.append("no part verdicts were produced")
        if repeats:
            problems.append(f"parts judged twice: {repeats[:5]}")
        if counts.get("not_inspected"):
            problems.append(f"{counts['not_inspected']} parts not inspected")
        for r in verdicts:
            expected_fail = r["part"] % 5 == 0   # config.360.yaml places a demo defect on every 5th part
            if (r["verdict"] == "fail") != expected_fail:
                problems.append(f"part {r['part']} verdict {r['verdict']} (demo defect: {expected_fail})")

        check = subprocess.run([sys.executable, "-c", DASHBOARD_CHECK.format(app=ROOT / "src/dashboard/app.py")],
                               cwd=work, env=env, capture_output=True, text=True, timeout=150)
        if check.returncode != 0:
            problems.append("dashboard: " + check.stderr.strip().splitlines()[-1][:200])
        else:
            print(check.stdout.strip().splitlines()[-1])

    for problem in problems:
        print("FAIL:", problem)
    print("SMOKE TEST PASSED" if not problems else f"SMOKE TEST FAILED ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
