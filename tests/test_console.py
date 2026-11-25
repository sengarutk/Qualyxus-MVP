from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.camera.view_store import ViewStore
from src.console.data import ConsoleData
from src.console.server import create_app
from src.inference.audit_log import InspectionAuditLog
from src.process.models import FAIL, PASS, InspectionRecord
from src.process.service import ProcessIntelligenceService
from src.process.simulator import InjectionMoldingSimulator

FRAME = np.full((24, 32, 3), 90, np.uint8)
VIEWS = ("view_top", "view_side")


@pytest.fixture
def station(tmp_path):
    """150 cycles; the check ring starts wearing hard at cycle 121, and the last part fails."""
    cfg = {"process": {"db_path": str(tmp_path / "p.db"), "models_dir": str(tmp_path / "m"), "source": "simulated"},
           "inspection": {"audit": {"dir": str(tmp_path / "audit"), "key_file": str(tmp_path / "k")},
                          "view_store": {"dir": str(tmp_path / "views")}, "view_timeout_s": 0.0},
           "console": {"status_path": str(tmp_path / "status.json")},
           "cameras": [{"id": v, "required_view": True} for v in VIEWS]}
    service = ProcessIntelligenceService(cfg["process"])
    audit = InspectionAuditLog(tmp_path / "audit", tmp_path / "k")
    views = ViewStore(tmp_path / "views", keep=5)
    sim = InjectionMoldingSimulator(seed=11)
    sim.load_scenario({"check_ring_wear": 0.02})
    start, cycles = time.time() - 600, []
    for i in range(150):
        if i == 120:
            sim.drift_per_cycle = {"check_ring_wear": 0.006}
        cycle, _ = sim.next_cycle()
        cycle.end_ts = start + i * 3
        service.on_cycle(cycle)
        cycles.append(cycle)
        for view in VIEWS:
            failed = i == 149 and view == "view_side"
            service.store.save_inspection(InspectionRecord("", view, cycle.end_ts + 1, FAIL if failed else PASS,
                                                           defect_type="sink_mark" if failed else None,
                                                           anomaly_score=1.6 if failed else 0.5,
                                                           cycle_id=cycle.cycle_id, alignment_confidence=0.9))
            views.save(cycle.cycle_id, view, FRAME, FRAME if failed else None)
        audit.append({"ts": cycle.end_ts, "record": "part_verdict", "cycle_id": cycle.cycle_id,
                      "verdict": "fail" if i == 149 else "pass"})
    return cfg, cycles, tmp_path


def _write_status(tmp_path, **plc):
    workers = {v: {"failed": False, "capture_alive": True, "parts_inspected": 3} for v in VIEWS}
    (tmp_path / "status.json").write_text(json.dumps({"timestamp": time.time(), "workers": workers, "plc": plc}))


def test_counts_match_what_was_recorded(station):
    cfg, _, tmp_path = station
    _write_status(tmp_path, simulated=True, hold_reason=None)
    state = ConsoleData(cfg).state()
    assert state["online"] and state["simulated"]
    assert (state["kpi"]["parts"], state["kpi"]["pass"], state["kpi"]["fail"]) == (150, 149, 1)
    assert state["audit"]["ok"] and state["audit"]["records"] == 150
    assert all(c["online"] for c in state["cameras"])


def test_rejected_part_traces_to_the_signal_that_moved(station):
    cfg, cycles, _ = station
    detail = ConsoleData(cfg).part_detail(cycles[-1].cycle_id)
    assert detail["verdict"] == "fail" and detail["defects"] == ["sink_mark"]
    assert [v["verdict"] for v in detail["views"]] == ["pass", "fail"]
    assert "heat" in detail["views"][1] and "heat" not in detail["views"][0]
    assert detail["deviations"][0]["points_to"] == "Check ring"
    assert {c["name"] for c in detail["curves"]} >= {"hold_pressure", "cavity_pressure"}
    assert detail["audit"][0]["verdict"] == "fail"


def test_healthy_part_gets_no_invented_cause(station):
    cfg, cycles, _ = station
    detail = ConsoleData(cfg).part_detail(cycles[60].cycle_id)
    assert not any(d["points_to"] for d in detail["deviations"])


def test_worn_component_raises_an_alarm(station):
    cfg, _, tmp_path = station
    _write_status(tmp_path, hold_reason=None)
    titles = [a["title"] for a in ConsoleData(cfg).state()["alarms"]]
    assert any(t.startswith("Check ring") for t in titles)


def test_stopped_runtime_and_held_heartbeat_are_never_shown_as_healthy(station):
    cfg, _, tmp_path = station
    assert ConsoleData(cfg).state()["alarms"][0]["title"] == "Inspection runtime is offline"
    _write_status(tmp_path, hold_reason="no frames from ['view_top']")
    state = ConsoleData(cfg).state()
    assert any(a["title"] == "PLC heartbeat held" for a in state["alarms"])


def test_broken_audit_trail_is_an_alarm(station):
    cfg, _, tmp_path = station
    log = next((tmp_path / "audit").glob("*.jsonl"))
    log.write_text(log.read_text().replace('"verdict": "pass"', '"verdict": "fail"', 1))
    assert any(a["title"] == "Audit trail broken" for a in ConsoleData(cfg).state()["alarms"])


def test_control_chart_flags_the_drift_and_not_the_baseline(station):
    cfg, _, _ = station
    chart = ConsoleData(cfg).spc("hold_pressure.active_median")
    beyond = [i for i, p in enumerate(chart["points"]) if p["rule"] == "beyond 3σ"]
    assert beyond and len([i for i in beyond if i < 100]) <= 5 and beyond[-1] == 149
    assert chart["points"][-1]["failed"]
    assert ConsoleData(cfg).spc("no_such_signal") is None


def test_empty_station_shows_nothing_rather_than_zeros_that_look_real(tmp_path):
    data = ConsoleData({"process": {"db_path": str(tmp_path / "none.db")},
                        "inspection": {"audit": {"dir": str(tmp_path / "a")}}, "console": {"status_path": "/nope"}})
    state = data.state()
    assert state["kpi"]["yield"] is None and state["audit"]["ok"] is None and state["health"] == []
    assert data.quality()["pareto"] == [] and data.risk() == [] and data.part_detail("x") is None


def test_server_is_read_only_and_cannot_be_walked(station):
    cfg, cycles, _ = station
    client = TestClient(create_app(cfg))
    page = client.get("/")
    assert page.status_code == 200 and "default-src 'self'" in page.headers["content-security-policy"]
    assert "http://" not in page.text and "https://" not in page.text          # works with no internet
    for path in ("state", "parts", "spc", "risk", "quality", "audit", "prescriptions", f"part/{cycles[-1].cycle_id}"):
        assert client.get(f"/api/{path}").status_code == 200, path
        assert client.post(f"/api/{path}").status_code == 405, path
    view = ConsoleData(cfg).part_detail(cycles[-1].cycle_id)["views"][0]["image"]
    assert client.get(view).headers["content-type"] == "image/jpeg"
    for bad in ("/img/..%2f..%2fetc/passwd", "/img/x/..%2fp.db", "/static/server.py", "/img/a/b.png"):
        assert client.get(bad).status_code == 404, bad


def test_token_protects_every_data_route(station, monkeypatch):
    cfg, _, _ = station
    monkeypatch.setenv("QUALYXUS_API_TOKEN", "s3cret")
    client = TestClient(create_app(cfg))
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    client.get("/?token=s3cret")
    assert client.get("/api/state").status_code == 401        # never from the address bar


def test_network_bind_needs_accounts_and_https(tmp_path, monkeypatch):
    from src.console import server
    from src.operations.auth import UserStore

    config = tmp_path / "config.yaml"
    config.write_text(f"console: {{users_file: {tmp_path}/users.json}}\ncameras: []\n")
    monkeypatch.setenv("CONFIG_PATH", str(config))
    started = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: started.append(k))
    for args in (["--host", "0.0.0.0"], ["--host", "0.0.0.0", "--tls-cert", "c", "--tls-key", "k"]):
        with pytest.raises(SystemExit):
            server.main(args)
    UserStore(tmp_path / "users.json").add("asha", "long enough password", "QA")
    with pytest.raises(SystemExit):
        server.main(["--host", "0.0.0.0"])
    server.main(["--host", "0.0.0.0", "--tls-cert", "c", "--tls-key", "k"])
    server.main([])
    assert [s["host"] for s in started] == ["0.0.0.0", "127.0.0.1"] and started[0]["ssl_certfile"] == "c"


def test_view_store_keeps_only_the_newest_parts(tmp_path):
    store = ViewStore(tmp_path / "v", keep=3)
    for i in range(8):
        store.save(f"m-{i}#0", "view/../top", FRAME)
    folders = sorted(p.name for p in (tmp_path / "v").iterdir())
    assert len(folders) == 3 and "m-7_0" in folders
    assert cv2.imread(str(next((tmp_path / "v" / "m-7_0").glob("*.jpg")))) is not None


def test_scenario_rejects_unknown_state():
    with pytest.raises(ValueError):
        InjectionMoldingSimulator().load_scenario({"nonsense": 1.0})
