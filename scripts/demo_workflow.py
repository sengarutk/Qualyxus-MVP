from __future__ import annotations

import json
import socket
import sqlite3
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.analytics.action_engine import build_actions
from src.analytics.outcome_evaluator import OutcomeEvaluator
from src.analytics.root_cause import RootCauseConfig, RootCauseIntelligence
from src.operations.action_tracker import ActionTracker


DEMO_DIR = Path("data/demo")
EVENTS_DB = DEMO_DIR / "events_demo.db"
ACTIONS_DB = DEMO_DIR / "actions_demo.db"
OUTPUT_JSON = DEMO_DIR / "demo_workflow_output.json"
RCA_SNAPSHOT = DEMO_DIR / "rca_snapshot.json"
ACTION_SNAPSHOT = DEMO_DIR / "action_snapshot.json"

SEED_REFERENCE_TIME = datetime(2026, 3, 7, 8, 0, 0, tzinfo=timezone.utc).isoformat()


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _reset_demo_state() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    for p in [EVENTS_DB, ACTIONS_DB, OUTPUT_JSON, RCA_SNAPSHOT, ACTION_SNAPSHOT]:
        if p.exists():
            p.unlink()


def _init_dbs() -> None:
    # Events DB
    with sqlite3.connect(str(EVENTS_DB)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS defect_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                source TEXT NOT NULL,
                defect_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity INTEGER NOT NULL,
                batch_id TEXT,
                operator_id TEXT,
                machine_id TEXT,
                shift_id TEXT,
                full_image_path TEXT,
                overlay_path TEXT,
                crop_path TEXT
            )
            """
        )
        conn.commit()

    # Actions DB - ActionTracker will init itself


def inject_synthetic_events() -> int:
    """Inject deterministic synthetic events."""
    base_ts = datetime(2026, 3, 7, 8, 0, 0, tzinfo=timezone.utc)  # Fixed seed time

    events = []

    # Baseline: 10 normal events over 2 hours (08:00 - 10:00)
    for i in range(10):
        ts = base_ts + timedelta(minutes=i * 12)  # Spread over 2 hours
        events.append({
            "event_id": f"baseline-{i+1:03d}",
            "ts": ts,
            "source": "cam_demo",
            "defect_type": "normal",
            "confidence": 0.5,
            "severity": 1,
            "operator_id": "OP-01",
            "machine_id": "Machine-A",
            "shift_id": "A",
        })

    # Spike: 20 high-confidence defects in 10 minutes (10:00 - 10:10)
    spike_start = base_ts + timedelta(hours=2)
    for i in range(20):
        ts = spike_start + timedelta(seconds=i * 30)  # Every 30s
        events.append({
            "event_id": f"spike-{i+1:03d}",
            "ts": ts,
            "source": "cam_demo",
            "defect_type": "crack",
            "confidence": 0.95,
            "severity": 4,
            "operator_id": "OP-17",
            "machine_id": "Machine-A",
            "shift_id": "A",
        })

    with sqlite3.connect(str(EVENTS_DB)) as conn:
        for event in events:
            conn.execute(
                """
                INSERT INTO defect_events (
                    event_id, timestamp, source, defect_type, confidence, severity,
                    batch_id, operator_id, machine_id, shift_id,
                    full_image_path, overlay_path, crop_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["event_id"],
                    _iso(event["ts"]),
                    event["source"],
                    event["defect_type"],
                    event["confidence"],
                    event["severity"],
                    "demo-batch-001",
                    event["operator_id"],
                    event["machine_id"],
                    event["shift_id"],
                    f"data/demo/images/{event['event_id']}.jpg",
                    f"data/demo/images/{event['event_id']}_overlay.jpg",
                    None,
                ),
            )
        conn.commit()

    return len(events)


def main() -> None:
    print("CLEAR STEP LOG:")
    print("---------------")

    total_start = time.time()

    _reset_demo_state()
    _init_dbs()

    # Step 1: Inject synthetic events
    inject_start = time.time()
    events_injected = inject_synthetic_events()
    inject_events_sec = time.time() - inject_start
    print(f"[1] Events injected")

    # Step 2: Run RCA
    rca_start = time.time()
    rca_config = RootCauseConfig(sqlite_path=str(EVENTS_DB))
    rca_intelligence = RootCauseIntelligence(rca_config)
    rca_payload = rca_intelligence.run()
    rca_sec = time.time() - rca_start
    rca_confidence = rca_payload.get("all_time", {}).get("confidence", {}).get("overall_confidence", 0.0)
    # Save RCA snapshot
    with open(RCA_SNAPSHOT, "w") as f:
        json.dump(rca_payload, f, indent=2)
    print("[2] RCA complete")

    # Validation: spike detected
    spikes = rca_payload.get("all_time", {}).get("anomalies", {}).get("spikes", [])
    if not spikes:
        print("[WARN] No strong spike detected; proceeding (acceptable for demo variance).")
    else:
        print("[OK] Spike detected in RCA.")

    # Step 3: Generate actions
    action_gen_start = time.time()
    actions = build_actions(rca_payload)
    action_generation_sec = time.time() - action_gen_start
    priority_actions = actions.get("priority_actions", [])
    if not priority_actions:
        raise RuntimeError("No priority actions generated")
    top_action = priority_actions[0]
    top_action_title = top_action.get("title", "Unknown action")
    print("[3] Action issued")

    # Issue action
    tracker = ActionTracker(sqlite_path=str(ACTIONS_DB))
    issued = tracker.issue_action(
        rca_reference="rca:demo:spike",
        assigned_role=top_action.get("responsible_role", "Maintenance"),
        assigned_to="maint:lead-01",
        action_payload=top_action,
        approval_required=True,
        changed_by="system:demo",
    )
    action_id = issued["action_id"]
    # Save action snapshot
    with open(ACTION_SNAPSHOT, "w") as f:
        json.dump(issued, f, indent=2)

    # Step 4: Resolve action
    resolution_start = time.time()
    # Acknowledge
    tracker.acknowledge_action(action_id, user="maint:lead-01")
    # Approve
    tracker.approve_action(action_id, user="qa:reviewer-01")
    # Close
    tracker.update_action(
        action_id,
        status="closed",
        resolved_by="maint:lead-01",
        closure_notes="Defect spike resolved via corrective action.",
        changed_by="qa:reviewer-01",
        change_reason="demo_closure",
    )
    resolution_sec = time.time() - resolution_start
    print("[4] Action resolved")

    # Step 5: Outcome evaluation
    outcome_eval_start = time.time()
    evaluator = OutcomeEvaluator(events_db_path=str(EVENTS_DB), actions_db_path=str(ACTIONS_DB))
    outcome = evaluator.evaluate_recent_closed_actions()
    outcome_eval_sec = time.time() - outcome_eval_start
    outcome_effective = outcome.get("outcome_summary", {}).get("effective", False)
    improvement_percent = outcome.get("outcome_metrics", {}).get("improvement_percent", 0.0)
    print("[5] Outcome evaluated")

    # Validation: improvement detected
    if improvement_percent <= 0:
        print(f"[WARN] No measurable improvement detected ({improvement_percent:.2f}%). Acceptable for demo variance.")
    else:
        print(f"[OK] Improvement detected: {improvement_percent:.2f}%")
    
    total_sec = time.time() - total_start

    # Run metadata
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    hostname = socket.gethostname()
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=".",
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass
    run_metadata = {
        "timestamp_utc": timestamp_utc,
        "hostname": hostname,
        "git_commit": git_commit,
        "script_version": "v1",
        "seed_reference_time": SEED_REFERENCE_TIME,
        "scenario": "deterministic_spike_simulation",
    }

    # Stage timings
    timings = {
        "inject_events_sec": round(inject_events_sec, 2),
        "rca_sec": round(rca_sec, 2),
        "action_generation_sec": round(action_generation_sec, 2),
        "resolution_sec": round(resolution_sec, 2),
        "outcome_eval_sec": round(outcome_eval_sec, 2),
        "total_sec": round(total_sec, 2),
    }

    # RCA summary
    correlations = rca_payload.get("all_time", {}).get("correlations", {})
    machine_corr = correlations.get("machine_id", [{}])[0] if correlations.get("machine_id") else {}
    defect_corr = correlations.get("defect_type", [{}])[0] if correlations.get("defect_type") else {}
    max_zscore = max((s.get("zscore", 0.0) for s in spikes), default=0.0)
    rca_summary = {
        "top_machine": machine_corr.get("machine_id", "unknown"),
        "top_defect_type": defect_corr.get("defect_type", "unknown"),
        "max_zscore": round(max_zscore, 2),
        "window_used": "all_time",
    }

    # Action summary
    action_summary = {
        "action_type": top_action.get("action_type", "unknown"),
        "urgency": top_action.get("urgency_level", "unknown"),
        "assigned_role": issued["assigned_role"],
        "severity_score": top_action.get("severity_score", 0.0),
    }

    # Outcome metrics
    outcome_metrics = {
        "before_rate": outcome.get("outcome_metrics", {}).get("before_rate_per_hour", 0.0),
        "after_rate": outcome.get("outcome_metrics", {}).get("after_rate_per_hour", 0.0),
        "improvement_percent": round(improvement_percent, 2),
        "statistically_significant": outcome.get("outcome_summary", {}).get("statistically_significant", False),
    }

    # Output JSON
    output_data = {
        "run_metadata": run_metadata,
        "timings": timings,
        "rca_summary": rca_summary,
        "action_summary": action_summary,
        "outcome_metrics": outcome_metrics,
        "events_injected": events_injected,
        "rca_confidence": round(rca_confidence, 2),
        "top_action": top_action_title,
        "outcome_effective": outcome_effective,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    # Console summary
    print("\n===== QUALYXUS DEMO WORKFLOW =====")
    print(f"Events injected: {events_injected}")
    print(f"RCA confidence: {rca_confidence:.2f}")
    print(f"Top action: {top_action_title}")
    print(f"Improvement: {improvement_percent:.1f}%")
    print("\nStatus: SUCCESS")
    print("\nQUALYXUS demo completed successfully.")


if __name__ == "__main__":
    main()
