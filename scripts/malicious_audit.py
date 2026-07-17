from __future__ import annotations

"""Malicious end-to-end MVP stress test & logic validation.

Run:
  python scripts/malicious_audit.py
"""

import sqlite3
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


def _banner(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _bad(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_hash_collision() -> None:
    _banner("1A. Audit Hash Collision Check")
    from src.logging.event_logger import EventLogger
    from src.logging.schema import DefectEvent

    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "events.db")
        csv = str(Path(td) / "events.csv")
        logger = EventLogger(sqlite_path=db, csv_path=csv)
        base_ts = datetime(2026, 1, 1, 0, 0, 0)

        hashes: list[str] = []
        for i in range(100):
            ev = DefectEvent(
                event_id=logger.new_event_id(),
                timestamp=base_ts + timedelta(microseconds=i % 3),
                source="ipcam_1",
                defect_type="surface scratch",
                confidence=0.9,
                severity=2,
                batch_id=None,
                operator_id=None,
                machine_id="m1",
                shift_id="s1",
                line_id="l1",
                station_id="st1",
                temperature=78.1,
                vibration=0.01,
                speed=1.0,
                lot_id="lot1",
                full_image_path="/tmp/x.jpg",
                overlay_path=None,
                crop_path=None,
            )
            logger.log_event(ev)
            _assert(ev.audit_hash is not None, "missing audit_hash")
            hashes.append(ev.audit_hash)

        _assert(len(set(hashes)) == 100, f"hash collision detected: unique={len(set(hashes))}/100")
        h1 = logger._generate_audit_hash(ev)
        h2 = logger._generate_audit_hash(ev)
        _assert(h1 == h2, "audit hash is not deterministic")
        _ok("100/100 unique SHA-256 hashes; deterministic per event")


def test_vlm_context_overflow() -> None:
    _banner("1B. VLM Context Overflow")
    from src.analytics.vlm_narrator import VLMNarrator

    narrator = VLMNarrator({"enabled": False})
    big = "X" * 10_000
    ctx = {
        "machine_id": "m1",
        "shift_id": "s1",
        "temperature": 80.0,
        "vibration": 0.2,
        "speed": 1.2,
        "line_id": "l1",
        "station_id": "st1",
        "telemetry_context": big,
        "notes": big,
    }
    summary = narrator._telemetry_summary(ctx)
    _assert(len(summary) < 2000, f"telemetry summary too large: {len(summary)} chars")
    _ok(f"telemetry summary bounded (len={len(summary)})")


def test_sql_injection() -> None:
    _banner("1C. SQL Injection Check")
    from src.logging.event_logger import EventLogger
    from src.logging.schema import DefectEvent

    with tempfile.TemporaryDirectory() as td:
        db = str(Path(td) / "events.db")
        csv = str(Path(td) / "events.csv")
        logger = EventLogger(sqlite_path=db, csv_path=csv)
        injection = "defect'); DROP TABLE defect_events; --"
        ev = DefectEvent(
            event_id=logger.new_event_id(),
            timestamp=datetime.utcnow(),
            source="ipcam_1",
            defect_type=injection,
            confidence=0.9,
            severity=2,
            batch_id=None,
            operator_id=None,
            machine_id=None,
            shift_id=None,
            line_id=None,
            station_id=None,
            temperature=None,
            vibration=None,
            speed=None,
            lot_id=None,
            full_image_path="/tmp/x.jpg",
            overlay_path=None,
            crop_path=None,
        )
        logger.log_event(ev)
        with sqlite3.connect(db) as conn:
            c = conn.execute("SELECT COUNT(*) FROM defect_events").fetchone()[0]
        _assert(c == 1, f"expected 1 row, got {c}")
        _ok("SQL injection payload stored as data; defect_events table intact")


@dataclass
class _FakeVitals:
    cpu_temperature_c: float
    gpu_temperature_c: float
    vram_percent: float = 0.0
    ram_percent: float = 0.0
    timestamp: float = 0.0


def test_governor_thermal_jump_and_hysteresis() -> None:
    _banner("2. Self-Healing Governor Stress Test")
    from src.config.config import Config
    from src.logging.schema import SystemState
    from src.system.resource_monitor import ResourceGovernor

    cfg = Config.load(
        overrides={
            "resource_management": {
                "temp_thresholds": {"warning": 78, "critical": 85},
                "warning_hold_seconds": 0,
                "critical_hold_seconds": 0,
                "warning_cooldown_seconds": 5,
                "critical_cooldown_seconds": 5,
                "emergency_temp_c": 90,
            }
        }
    )

    gov = ResourceGovernor(config=cfg)
    vitals = _FakeVitals(cpu_temperature_c=30.0, gpu_temperature_c=30.0, timestamp=time.time())
    gov.get_vitals = lambda: vitals  # type: ignore

    called = {"n": 0}
    gov.set_emergency_callback(lambda v: called.__setitem__("n", called["n"] + 1))

    _assert(gov.get_policy_state() == SystemState.SAFE, "expected SAFE baseline")
    vitals.cpu_temperature_c = 90.0
    state = gov.get_policy_state()
    _assert(state == SystemState.EMERGENCY, f"expected EMERGENCY, got {state}")
    _assert(called["n"] == 1, f"emergency callback not triggered once (n={called['n']})")
    _ok("Thermal policy jump: SAFE -> EMERGENCY immediately")

    gov2 = ResourceGovernor(config=cfg)
    vitals2 = _FakeVitals(cpu_temperature_c=77.0, gpu_temperature_c=0.0, timestamp=time.time())
    gov2.get_vitals = lambda: vitals2  # type: ignore
    states: list[str] = []
    for i in range(20):
        vitals2.cpu_temperature_c = 77.0 if i % 2 == 0 else 79.0
        states.append(gov2.get_policy_state().value)
        time.sleep(0.05)
    transitions = sum(1 for a, b in zip(states, states[1:]) if a != b)
    _assert(transitions <= 3, f"status flapping detected: transitions={transitions}, states={states}")
    _ok(f"Hysteresis validated (transitions={transitions})")


def test_uns_replay_fifo() -> None:
    _banner("3. Network Chaos Replay Test")
    from src.telemetry.uns_client import UNSClient

    with tempfile.TemporaryDirectory() as td:
        replay_path = str(Path(td) / "buf.jsonl")
        client = UNSClient(
            {
                "agentic_enabled": True,
                "replay_buffer_path": replay_path,
                "replay_buffer_limit": 6000,
                "max_queue_size": 10,
            }
        )

        # Intentionally generate many actions while disconnected. We keep stdout clean.
        import logging

        logging.getLogger("src.telemetry.uns_client").setLevel(logging.ERROR)
        for i in range(500):
            client.publish_signal("ACTION", {"seq": i, "ts": time.time()})

        lines = Path(replay_path).read_text(encoding="utf-8").splitlines()
        _assert(len(lines) >= 490, f"expected ~500 buffered messages, got {len(lines)}")
        first = lines[0]
        # Ack a specific message id (new QoS1 hardening path).
        import json as _json

        first_obj = _json.loads(first)
        _assert(isinstance(first_obj, dict) and first_obj.get("message_id"), "buffer entry missing message_id")
        client._ack_replay_message(str(first_obj.get("message_id")))
        lines2 = Path(replay_path).read_text(encoding="utf-8").splitlines()
        _assert(lines2 and lines2[0] != first, "FIFO prune failed: first line unchanged")
        _ok("Replay buffer persists to disk and prunes FIFO")


def test_camera_pipeline_sabotage() -> None:
    _banner("4. GStreamer & Hardware Fallback")
    from src.camera.camera_manager import CameraManager

    mgr = CameraManager(
        config_data={
            "camera_settings": {"use_gstreamer": True, "hardware_platform": "nvidia"},
            "cameras": [{"id": "cam1", "source": "rtsp://invalid", "enabled": False}],
        }
    )
    start = time.time()
    cap, mode = mgr._open_camera("rtsp://invalid !!! MALFORMED", width=320, height=240, fps=10)
    elapsed = time.time() - start
    _assert(elapsed <= 6.0, f"fallback exceeded timeout: {elapsed:.2f}s")
    _assert(mode in {"OpenCV-SW", "GStreamer-HW"}, f"unexpected ingest mode: {mode}")
    _ok(f"Camera open fallback completed (mode={mode}, elapsed={elapsed:.2f}s)")
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass


def test_iso_mapping() -> None:
    _banner("5. ISO-9001 Compliance Accuracy")
    from src.compliance.iso_mapper import ISOMapper
    from src.logging.schema import DefectEvent

    base = dict(
        event_id="e",
        timestamp=datetime.utcnow(),
        source="ipcam_1",
        confidence=0.9,
        severity=2,
        batch_id=None,
        operator_id=None,
        machine_id=None,
        shift_id=None,
        line_id=None,
        station_id=None,
        temperature=None,
        vibration=None,
        speed=None,
        lot_id=None,
        full_image_path="/tmp/x.jpg",
        overlay_path=None,
        crop_path=None,
    )
    e1 = DefectEvent(defect_type="Missing Component", **base)
    e2 = DefectEvent(defect_type="Surface Scratch", **base)
    _assert(ISOMapper.clause_for_event(e1) == "Clause 8.7 Non-conforming Outputs", "Missing Component mapping wrong")
    _assert(ISOMapper.clause_for_event(e2) == "Clause 8.5.1 Control of Production", "Surface Scratch mapping wrong")
    _ok("ISO mappings validated")


def main() -> int:
    failures: list[tuple[str, str]] = []
    tests = [
        test_hash_collision,
        test_vlm_context_overflow,
        test_sql_injection,
        test_governor_thermal_jump_and_hysteresis,
        test_uns_replay_fifo,
        test_camera_pipeline_sabotage,
        test_iso_mapping,
    ]

    for fn in tests:
        try:
            fn()
        except Exception as exc:
            failures.append((fn.__name__, str(exc)))
            _bad(f"{fn.__name__}: {exc}")

    _banner("SUMMARY")
    if not failures:
        _ok("All malicious-audit checks passed")
        return 0
    for name, msg in failures:
        _bad(f"{name}: {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
