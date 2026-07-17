from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    details: str
    execution_time_seconds: float
    failure_type: str
    recovered_cleanly: bool


def _base_config() -> dict:
    return {
        "paths": {
            "data_dir": "data",
            "models_dir": "models/checkpoints",
        },
        "logging": {
            "sqlite_path": "data/events/events.db",
            "csv_path": "data/events/events.csv",
        },
        "inference": {
            "conf": 0.35,
            "skip_frames": 1,
        },
        "camera_manager": {
            "monitor_interval": 0.5,
            "health_timeout": 3.0,
            "restart_backoff": 0.5,
        },
        "watchdog": {
            "heartbeat_timeout": 2.5,
            "frame_drop_threshold": 150,
            "queue_overflow_streak_threshold": 8,
            "base_backoff": 0.5,
            "max_backoff": 4.0,
        },
        "disk_monitor": {
            "watch_path": "data",
            "interval_seconds": 0.5,
            "warning_threshold": 15.0,
            "critical_threshold": 10.0,
        },
        "runtime_metrics": {
            "output_path": "data/runtime/runtime_metrics.json",
            "interval_seconds": 1.0,
        },
        "camera": {
            "fps": 10,
            "width": 320,
            "height": 240,
            "reconnect_delay": 1.0,
            "save_every": 2,
            "frame_queue_size": 4,
            "event_queue_size": 8,
            "raw_dir": "data/live/raw",
            "defect_dir": "data/live/defects",
        },
        "cameras": [
            {
                "id": "cam_test",
                "source": "synthetic://test",
                "enabled": True,
                "mock_inference": True,
                "mock_event_every": 1,
                "independent_logs": True,
            }
        ],
    }


def _run_manager(cfg: dict, run_seconds: float = 6.0) -> dict:
    try:
        from src.camera.camera_manager import CameraManager
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "CameraManager dependencies are missing (likely opencv-python). "
            "Install runtime dependencies before executing resilience tests."
        ) from exc

    manager = CameraManager(config_data=cfg)
    manager.start()
    try:
        time.sleep(run_seconds)
        return manager.get_status_snapshot()
    finally:
        manager.stop()


def _scenario_camera_drop() -> ScenarioResult:
    cfg = _base_config()
    cfg["cameras"][0]["id"] = "cam_drop"
    cfg["cameras"][0]["source"] = "synthetic://drop"
    cfg["cameras"][0]["synthetic_fail_after_frames"] = 10

    start_time = time.time()
    snapshot = _run_manager(cfg, run_seconds=8.0)
    recovery_time = time.time() - start_time

    info = snapshot["cam_drop"]
    passed = info["restarts"] >= 1 and info["capture_alive"]
    details = f"restarts={info['restarts']}, capture_alive={info['capture_alive']}"
    return ScenarioResult("camera_drop", passed, details, recovery_time, "camera_disconnect", passed)


def _scenario_db_lock() -> ScenarioResult:
    cfg = _base_config()
    cfg["cameras"][0]["id"] = "cam_db_lock"
    cfg["cameras"][0]["source"] = "synthetic://db-lock"

    os.environ["SIMULATE_DB_LOCK"] = "1"
    try:
        start_time = time.time()
        snapshot = _run_manager(cfg, run_seconds=6.0)
        recovery_time = time.time() - start_time
    finally:
        os.environ.pop("SIMULATE_DB_LOCK", None)

    info = snapshot["cam_db_lock"]
    # Under DB lock, logger should keep running even if writes fail.
    passed = info["logger_alive"] and info["frames_captured"] > 0
    details = (
        f"logger_alive={info['logger_alive']}, frames_captured={info['frames_captured']}, "
        f"events_logged={info['events_logged']}"
    )
    return ScenarioResult("db_lock", passed, details, recovery_time, "database_lock", passed)


def _scenario_disk_full() -> ScenarioResult:
    cfg = _base_config()
    cfg["cameras"][0]["id"] = "cam_disk_full"
    cfg["cameras"][0]["source"] = "synthetic://disk"

    os.environ["SIMULATED_DISK_FREE_PERCENT"] = "8"
    from src.camera.camera_manager import CameraManager

    start_time = time.time()
    manager = CameraManager(config_data=cfg)
    manager.start()
    try:
        time.sleep(4.0)
        policy = manager.disk_monitor.get_policy()
        snapshot = manager.get_status_snapshot()
        recovery_time = time.time() - start_time
    finally:
        manager.stop()
        os.environ.pop("SIMULATED_DISK_FREE_PERCENT", None)

    info = snapshot["cam_disk_full"]
    passed = (
        policy.level == "critical"
        and not policy.raw_enabled
        and not policy.defect_enabled
        and info["frames_captured"] > 0
    )
    details = (
        f"disk_level={policy.level}, raw_enabled={policy.raw_enabled}, "
        f"defect_enabled={policy.defect_enabled}, frames_captured={info['frames_captured']}"
    )
    return ScenarioResult("disk_full", passed, details, recovery_time, "disk_exhaustion", passed)


def _scenario_network_delay() -> ScenarioResult:
    """
    Simulate network delay/jitter by forcing sparse frame arrival.

    We approximate delayed network stream behavior with very low FPS and a short
    watchdog heartbeat timeout, expecting auto-recovery restarts.
    """
    cfg = _base_config()
    cfg["cameras"][0]["id"] = "cam_net_delay"
    cfg["cameras"][0]["source"] = "synthetic://net-delay"
    cfg["camera"]["fps"] = 1.0
    cfg["watchdog"]["heartbeat_timeout"] = 0.2

    start_time = time.time()
    snapshot = _run_manager(cfg, run_seconds=10.0)
    recovery_time = time.time() - start_time

    info = snapshot["cam_net_delay"]
    passed = info["restarts"] >= 1
    details = (
        f"restarts={info['restarts']}, frames_captured={info['frames_captured']}, "
        f"capture_alive={info['capture_alive']}"
    )
    return ScenarioResult("network_delay", passed, details, recovery_time, "network_latency", passed)


def main():
    scenarios = [
        _scenario_camera_drop,
        _scenario_db_lock,
        _scenario_disk_full,
        _scenario_network_delay,
    ]

    results: list[ScenarioResult] = []
    for fn in scenarios:
        print(f"\n=== Running scenario: {fn.__name__.replace('_scenario_', '')} ===")
        result = fn()
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.details} (execution: {result.execution_time_seconds:.2f}s)")

    # Sort results by scenario name for stable ordering
    results.sort(key=lambda r: r.name)

    # Compute summary
    total_scenarios = len(results)
    passed_count = sum(1 for r in results if r.passed)
    failed_count = total_scenarios - passed_count
    avg_execution_time = sum(r.execution_time_seconds for r in results) / total_scenarios

    # Print improved summary
    print("\nResilience Test Summary")
    print("-----------------------")
    print(f"Passed: {passed_count}/{total_scenarios}")
    print(f"Failed: {failed_count}/{total_scenarios}")
    print(f"Avg Execution Time: {avg_execution_time:.2f}s")

    failed = [r for r in results if not r.passed]
    if failed:
        raise AssertionError(f"Resilience test failed for: {[r.name for r in failed]}")

    print("\n✅ All resilience simulations passed")

    # Export reports
    _export_reports(results)


def _export_reports(results: list[ScenarioResult]):
    """Export test results to JSON and Markdown reports."""
    Path("reports").mkdir(exist_ok=True)

    # Compute summary
    total_scenarios = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total_scenarios - passed
    avg_execution_time_seconds = sum(r.execution_time_seconds for r in results) / total_scenarios

    # Run metadata
    run_timestamp = datetime.utcnow().isoformat()
    hostname = socket.gethostname()
    git_commit = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()
    except Exception:
        pass

    run_metadata = {
        "run_timestamp": run_timestamp,
        "hostname": hostname,
        "git_commit": git_commit,
    }

    summary = {
        "total_scenarios": total_scenarios,
        "passed": passed,
        "failed": failed,
        "avg_execution_time_seconds": round(avg_execution_time_seconds, 2),
    }

    # JSON export
    json_data = {
        "run_metadata": run_metadata,
        "summary": summary,
        "results": [
            {
                "scenario": r.name,
                "failure_type": r.failure_type,
                "outcome": "pass" if r.passed else "fail",
                "execution_time_seconds": round(r.execution_time_seconds, 2),
                "recovered_cleanly": r.recovered_cleanly,
            }
            for r in results
        ],
    }
    with open("reports/resilience_report.json", "w") as f:
        json.dump(json_data, f, indent=2)

    # Markdown export
    md_lines = [
        "# Resilience Test Report",
        "",
        "| Scenario | Failure Type | Outcome | Execution Time (s) | Clean Recovery |",
        "|----------|--------------|---------|---------------------|----------------|",
    ]
    for r in results:
        outcome = "pass" if r.passed else "fail"
        execution = f"{r.execution_time_seconds:.2f}"
        clean_recovery = "Yes" if r.recovered_cleanly else "No"
        md_lines.append(f"| {r.name} | {r.failure_type} | {outcome} | {execution} | {clean_recovery} |")
    md_lines.append("")

    with open("reports/resilience_report.md", "w") as f:
        f.write("\n".join(md_lines))


if __name__ == "__main__":
    main()
