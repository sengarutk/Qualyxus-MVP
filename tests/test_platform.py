"""Stage 4 of the audit: accounts, approvals, backup, health, alerts, uplink."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.console.server import COOKIE, create_app
from src.inference.audit_log import InspectionAuditLog, verify_chain
from src.inference.model_registry import VisionModelRegistry
from src.operations.action_tracker import ActionTracker
from src.operations.auth import LOCKOUT_S, MAX_FAILURES, UserStore
from src.process.models import ProcessCycle
from src.process.store import ProcessStore

PASSWORD = "long enough password"
HEADERS = {"X-Qualyxus": "1"}


@pytest.fixture
def site(tmp_path):
    cfg = {"process": {"db_path": str(tmp_path / "p.db"), "machine_id": "imm_1"},
           "inspection": {"registry_dir": str(tmp_path / "reg"), "review": {"dir": str(tmp_path / "review")},
                          "audit": {"dir": str(tmp_path / "audit"), "key_file": str(tmp_path / "audit.key")}},
           "console": {"users_file": str(tmp_path / "users.json"), "status_path": str(tmp_path / "status.json")},
           "cameras": [{"id": "top"}]}
    store = ProcessStore(tmp_path / "p.db")
    store.save_cycle(ProcessCycle("imm_1-1", "imm_1", 1, 1.0, 2.0), {"a": 1.0})
    users = UserStore(tmp_path / "users.json")
    for name, role in (("asha", "QA"), ("ravi", "maintenance"), ("om", "operator"), ("maya", "manager")):
        users.add(name, PASSWORD, role)
    app = create_app(cfg)

    def client(name=None):
        c = TestClient(app, headers=HEADERS)
        if name:
            assert c.post("/api/login", json={"name": name, "password": PASSWORD}).status_code == 200
        return c
    return cfg, tmp_path, users, client


def _records(tmp_path):
    return [json.loads(line) for f in sorted((tmp_path / "audit").glob("*.jsonl")) for line in f.read_text().splitlines()]


def _issue(tmp_path, approval=True):
    return ActionTracker(sqlite_path=str(tmp_path / "actions.db")).issue_action(
        rca_reference="cycle:x", assigned_role="production", approval_required=approval,
        action_payload={"title": "Adjust x", "action_type": "process_prescription", "explanation": "because"})["action_id"]


# --------------------------------------------------------------------------------- signing in

def test_everything_needs_a_sign_in_once_accounts_exist(site):
    _, _, _, client = site
    c = client()
    for path in ("/api/state", "/api/work", "/api/audit", "/metrics", "/api/parts"):
        assert c.get(path).status_code == 401, path
    assert c.get("/").status_code == 200 and c.get("/api/me").json()["accounts"] is True
    assert c.post("/api/login", json={"name": "asha", "password": "no"}).status_code == 401
    assert client("asha").get("/api/state").status_code == 200


def test_session_cookie_is_random_http_only_and_ends_on_sign_out(site):
    cfg, _, _, client = site
    c = client()
    header = c.post("/api/login", json={"name": "asha", "password": PASSWORD}).headers["set-cookie"].lower()
    assert "httponly" in header and "samesite=strict" in header and "asha" not in header
    secure = TestClient(create_app(cfg, secure_cookie=True), base_url="https://testserver")
    assert "secure" in secure.post("/api/login", json={"name": "asha", "password": PASSWORD}).headers["set-cookie"].lower()
    stolen = c.cookies.get(COOKIE)
    c.post("/api/logout")
    c.cookies.set(COOKIE, stolen)
    assert c.get("/api/state").status_code == 401
    assert client("asha").cookies.get(COOKIE) != stolen


def test_removed_or_demoted_person_loses_access_at_once(site):
    _, tmp_path, users, client = site
    c, action = client("asha"), _issue(tmp_path)
    users.add("asha", PASSWORD, "operator")
    assert c.post(f"/api/actions/{action}/approve", json={"reason": "fine"}).status_code == 403
    users.remove("asha")
    assert c.get("/api/state").status_code == 401


def test_one_address_cannot_lock_a_person_out_and_a_restart_does_not_unlock(tmp_path):
    store = UserStore(tmp_path / "u.json")
    store.add("asha", PASSWORD, "QA")
    for _ in range(MAX_FAILURES):
        store.authenticate("asha", "wrong", now=10.0, address="10.0.0.9")
    assert store.authenticate("asha", PASSWORD, now=11.0, address="10.0.0.9") is None
    assert UserStore(tmp_path / "u.json").authenticate("asha", PASSWORD, now=12.0, address="10.0.0.9") is None
    assert store.authenticate("asha", PASSWORD, now=12.0, address="10.0.0.5") == "QA"
    assert store.authenticate("asha", PASSWORD, now=11.0 + LOCKOUT_S, address="10.0.0.9") == "QA"


def test_trying_many_names_from_one_address_is_stopped(tmp_path):
    store = UserStore(tmp_path / "u.json")
    store.add("asha", PASSWORD, "QA")
    for i in range(30):
        store.authenticate(f"name{i}", "wrong", now=10.0, address="10.0.0.9")
    assert store.authenticate("asha", PASSWORD, now=11.0, address="10.0.0.9") is None


def test_without_accounts_the_console_reads_and_changes_nothing(site):
    cfg, tmp_path, users, _ = site
    for name in list(users.users()):
        users.remove(name)
    c, action = TestClient(create_app(cfg), headers=HEADERS), _issue(tmp_path)
    assert c.get("/api/state").status_code == 200
    assert c.post(f"/api/actions/{action}/approve", json={"reason": "fine"}).status_code == 401


def test_another_site_cannot_make_a_signed_in_browser_act(site):
    _, tmp_path, _, client = site
    c, action = client("asha"), _issue(tmp_path)
    assert c.post(f"/api/actions/{action}/approve", json={"reason": "fine"}, headers={"X-Qualyxus": ""}).status_code == 403


# ------------------------------------------------------------------------------- approvals (D5, D6)

def test_approver_and_closer_differ_and_reasons_are_needed(site):
    _, tmp_path, _, client = site
    asha, ravi, action = client("asha"), client("ravi"), _issue(tmp_path)
    assert client("om").post(f"/api/actions/{action}/start").status_code == 403
    assert ravi.post(f"/api/actions/{action}/approve", json={"reason": "fine"}).status_code == 403
    assert ravi.post(f"/api/actions/{action}/close", json={"reason": "done it"}).status_code == 409      # not approved
    assert asha.post(f"/api/actions/{action}/approve", json={"reason": " "}).status_code == 409           # no reason
    assert asha.post(f"/api/actions/{action}/approve", json={"reason": "within limits"}).status_code == 200
    assert asha.post(f"/api/actions/{action}/close", json={"reason": "done it"}).status_code == 409       # own approval
    assert ravi.post(f"/api/actions/{action}/start").status_code == 200
    assert ravi.post(f"/api/actions/{action}/close", json={"reason": "set to 640"}).status_code == 200
    assert ravi.post(f"/api/actions/{action}/close", json={"reason": "again"}).status_code == 409
    rows = [(r["by"], r["what"], r["reason"]) for r in _records(tmp_path)]
    assert rows == [("asha", "action_approve", "within limits"), ("ravi", "action_start", "started"),
                    ("ravi", "action_close", "set to 640")]
    assert verify_chain(tmp_path / "audit", tmp_path / "audit.key")["ok"]


def test_person_doing_the_work_cannot_approve_it(site):
    _, tmp_path, _, client = site
    asha, action = client("asha"), _issue(tmp_path)
    assert asha.post(f"/api/actions/{action}/start").status_code == 200
    assert asha.post(f"/api/actions/{action}/approve", json={"reason": "fine"}).status_code == 409
    assert client("maya").post(f"/api/actions/{action}/approve", json={"reason": "fine"}).status_code == 200


def test_tracker_itself_enforces_the_rules(tmp_path):
    action = _issue(tmp_path)
    tracker = ActionTracker(sqlite_path=str(tmp_path / "actions.db"))
    tracker.update_action(action, approved_by="asha", acknowledged_at="now", changed_by="asha")
    with pytest.raises(ValueError, match="what was done"):
        tracker.update_action(action, status="closed", changed_by="ravi")
    with pytest.raises(ValueError, match="cannot also close"):
        tracker.update_action(action, status="closed", changed_by="asha", closure_notes="done")
    tracker.update_action(action, status="closed", changed_by="ravi", closure_notes="done")


def test_nothing_changes_when_the_audit_trail_cannot_be_written(site):
    cfg, tmp_path, _, client = site
    asha, action = client("asha"), _issue(tmp_path)
    (tmp_path / "audit").write_text("a file where the folder should be")
    reply = asha.post(f"/api/actions/{action}/approve", json={"reason": "fine"})
    assert reply.status_code == 409 and "audit trail" in reply.json()["detail"]
    assert not ActionTracker(sqlite_path=str(tmp_path / "actions.db")).get_action(action).get("approved_by")


def _registry(tmp_path):
    reg = VisionModelRegistry(tmp_path / "reg")
    for i, kw in enumerate(({"activate": True}, {"shadow": True})):
        (tmp_path / f"m{i}.pt").write_text(str(i))
        reg.register("housing", "anomaly", tmp_path / f"m{i}.pt", **kw)
    return reg


def test_models_are_promoted_and_rolled_back_by_a_named_person(site):
    _, tmp_path, _, client = site
    reg = _registry(tmp_path)
    assert client("ravi").post("/api/models/housing/promote", json={"reason": "good"}).status_code == 403
    asha = client("asha")
    assert asha.post("/api/models/housing/rollback", json={"reason": "bad"}).status_code == 409
    assert asha.post("/api/models/housing/promote", json={"reason": "500 parts"}).json()["version"] == "v0002"
    assert asha.post("/api/models/housing/rollback", json={"reason": "rejects"}).json()["version"] == "v0001"
    assert asha.post("/api/models/..%2f..%2fx/rollback", json={"reason": "x y z"}).status_code == 404
    assert reg.manifest("housing", "anomaly")["history"][-1]["by"] == "asha"
    assert [r["what"] for r in _records(tmp_path)] == ["model_promote", "model_rollback"]


def test_retraining_request_is_decided_once(site):
    _, tmp_path, _, client = site
    request = _registry(tmp_path).request_retrain("housing", camera_id="top", reason="drift")
    asha = client("asha")
    assert asha.post(f"/api/requests/{request['id']}/approve", json={"reason": "new light"}).status_code == 200
    assert asha.post(f"/api/requests/{request['id']}/reject", json={"reason": "new light"}).status_code == 409


def test_command_line_model_changes_need_the_password(tmp_path, monkeypatch):
    from src.inference.model_registry import main

    reg = _registry(tmp_path)
    UserStore(tmp_path / "users.json").add("asha", PASSWORD, "QA")
    UserStore(tmp_path / "users.json").add("ravi", PASSWORD, "maintenance")
    config = tmp_path / "c.yaml"
    config.write_text(f"console: {{users_file: {tmp_path}/users.json}}\ninspection:\n  audit: {{dir: {tmp_path}/audit, "
                      f"key_file: {tmp_path}/audit.key}}\n")
    args = ["--root", str(reg.root), "--config", str(config), "promote", "--recipe", "housing", "--reason", "ok"]
    monkeypatch.setenv("QUALYXUS_PASSWORD", "wrong password")
    with pytest.raises(PermissionError):
        main([*args, "--by", "asha"])
    monkeypatch.setenv("QUALYXUS_PASSWORD", PASSWORD)
    with pytest.raises(PermissionError):
        main([*args, "--by", "ravi"])
    assert reg.manifest("housing", "anomaly")["active"] == "v0001"
    assert main([*args, "--by", "asha"]) == "v0002"
    assert [(r["by"], r["what"]) for r in _records(tmp_path)] == [("asha", "model_promote")]


def test_parts_are_reviewed_once_by_a_person(site):
    from src.inference.anomaly.monitoring import ReviewQueue

    _, tmp_path, _, client = site
    item = ReviewQueue(tmp_path / "review").add("top", np.zeros((40, 60, 3), np.uint8), normalized_score=1.0,
                                                verdict="fail", overlay=np.zeros((40, 60, 3), np.uint8)).stem
    asha = client("asha")
    (waiting,) = asha.get("/api/work").json()["review"]
    assert waiting["id"] == item
    assert asha.get(f"/review/top/{item}").status_code == 200 and asha.get(f"/review/top/{item}?overlay=1").status_code == 200
    for bad in ("/review/../x", "/review/top/..%2f..%2fusers", "/review/.lock/x"):
        assert asha.get(bad).status_code == 404
    assert client("ravi").post(f"/api/review/top/{item}", json={"verdict": "pass"}).status_code == 403
    assert asha.post(f"/api/review/top/{item}", json={"verdict": "maybe"}).status_code == 409
    assert asha.post(f"/api/review/top/{item}", json={"verdict": "pass"}).status_code == 200
    assert asha.post(f"/api/review/top/{item}", json={"verdict": "fail"}).status_code == 409
    meta = json.loads((tmp_path / "review" / "top" / f"{item}.json").read_text())
    assert meta["review"]["reviewer"] == "asha" and asha.get("/api/work").json()["review_waiting"] == 0


def test_baseline_is_reset_from_the_console(site):
    _, tmp_path, _, client = site
    ravi = client("ravi")
    assert ravi.post("/api/baseline/nope", json={"reason": "new ring"}).status_code == 404
    assert ravi.post("/api/baseline/imm_1", json={"reason": "new ring"}).status_code == 200
    assert ProcessStore(tmp_path / "p.db").baseline_start("imm_1") > 0


def test_metrics_are_prometheus_text(site):
    text = site[3]("om").get("/metrics").text
    assert "qualyxus_up 0" in text and 'qualyxus_parts{verdict="pass"} 0' in text and "None" not in text


# -------------------------------------------------------------------- audit with several writers

def test_runtime_and_console_can_write_the_same_trail(tmp_path):
    runtime, console = (InspectionAuditLog(tmp_path / "audit", tmp_path / "k") for _ in range(2))
    for i in range(6):
        (runtime if i % 2 else console).append({"n": i})
    assert verify_chain(tmp_path / "audit", tmp_path / "k") == {"ok": True, "records": 6, "head": runtime._last_hash}


def test_processes_writing_at_once_keep_one_chain(tmp_path):
    code = ("import sys; from src.inference.audit_log import InspectionAuditLog as L\n"
            "log = L(sys.argv[1], sys.argv[2])\nfor i in range(25): log.append({'n': i})")
    InspectionAuditLog(tmp_path / "audit", tmp_path / "k").append({"n": -1})
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).parents[1])}
    jobs = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "audit"), str(tmp_path / "k")], env=env)
            for _ in range(3)]
    assert [j.wait(60) for j in jobs] == [0, 0, 0]
    result = verify_chain(tmp_path / "audit", tmp_path / "k")
    assert result["ok"] and result["records"] == 76


# ---------------------------------------------------------------------------------- HTTPS (D1)

def test_certificate_serves_https(tmp_path):
    import uvicorn

    from src.system.tls import make_certificate

    cert, key = make_certificate(["station.local", "192.168.1.20"], tmp_path / "tls")
    assert oct(key.stat().st_mode)[-3:] == "600"
    server = uvicorn.Server(uvicorn.Config(create_app({"console": {"users_file": str(tmp_path / "u")}}), port=0,
                                           log_level="error", ssl_certfile=str(cert), ssl_keyfile=str(key)))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    port = server.servers[0].sockets[0].getsockname()[1]
    import httpx
    context = ssl.create_default_context(cafile=str(cert))
    assert httpx.get(f"https://127.0.0.1:{port}/api/me", verify=context).status_code == 200
    server.should_exit = True
    thread.join(5)


def test_api_is_not_opened_to_the_network_unprotected(monkeypatch):
    from src.api.pulse_server import PulseServer

    monkeypatch.delenv("QUALYXUS_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="TOKEN"):
        PulseServer("0.0.0.0", tls_cert="c", tls_key="k")
    monkeypatch.setenv("QUALYXUS_API_TOKEN", "t")
    with pytest.raises(ValueError, match="HTTPS"):
        PulseServer("0.0.0.0")
    assert PulseServer("0.0.0.0", tls_cert="c", tls_key="k").tls and PulseServer("127.0.0.1").tls == {}


# --------------------------------------------------------------------------------- backup (D10)

@pytest.fixture
def station(tmp_path, monkeypatch):
    home = tmp_path / "station"
    home.mkdir()
    monkeypatch.chdir(home)
    monkeypatch.setenv("CONFIG_PATH", "config.yaml")
    monkeypatch.setenv("QUALYXUS_MODEL_KEY_FILE", "secrets/model.key")
    Path("config.yaml").write_text("cameras: []\n")
    store = ProcessStore("data/events/process.db")
    for n in range(50):
        store.save_cycle(ProcessCycle(f"m-{n}", "m", n, n, n + 1), {"a": float(n)})
    log = InspectionAuditLog("data/audit", "secrets/audit.key")
    for n in range(5):
        log.append({"part": n})
    UserStore("secrets/users.json").add("asha", PASSWORD, "QA")
    Path("models").mkdir()
    Path("models/a.pt").write_bytes(b"model")
    Path("secrets/model.key").write_text("ab" * 32)
    cfg = {"inspection": {"anomaly_model": "models/a.pt", "audit": {"dir": "data/audit", "key_file": "secrets/audit.key"}}}
    return cfg, tmp_path, store


def test_backup_made_while_running_restores_everything(station):
    from src.system.backup import create, restore, verify

    cfg, tmp_path, store = station
    stop = threading.Event()

    def write():
        n = 1000
        while not stop.is_set():
            n += 1
            store.save_cycle(ProcessCycle(f"m-{n}", "m", n, n, n + 1), {"a": 1.0})
    thread = threading.Thread(target=write)
    thread.start()
    try:
        backup = create(cfg, tmp_path / "out", passphrase="a long passphrase")
    finally:
        stop.set()
        thread.join()
    assert verify(backup) == []
    into = restore(backup, tmp_path / "new", passphrase="a long passphrase")
    assert ProcessStore(into / "data/events/process.db").count_cycles() >= 50
    assert (into / "models/a.pt").read_bytes() == b"model" and (into / "config.yaml").exists()
    assert verify_chain(into / "data/audit", into / "secrets/audit.key")["records"] == 5
    assert UserStore(into / "secrets/users.json").authenticate("asha", PASSWORD) == "QA"
    InspectionAuditLog(into / "data/audit", into / "secrets/audit.key").append({"part": 6})
    assert verify_chain(into / "data/audit", into / "secrets/audit.key")["ok"]


def test_backup_never_holds_keys_in_the_clear(station):
    from src.system.backup import create, restore

    cfg, tmp_path, _ = station
    key = Path("secrets/audit.key").read_text()
    for passphrase in (None, "a long passphrase"):
        backup = create(cfg, tmp_path / f"out{passphrase is None}", passphrase=passphrase)
        assert not any(key.encode() in f.read_bytes() for f in backup.rglob("*") if f.is_file())
    with pytest.raises(ValueError, match="passphrase"):
        restore(backup, tmp_path / "x", passphrase="wrong")


def test_changed_backup_is_refused_and_restore_never_overwrites(station):
    from src.system.backup import create, restore, verify

    cfg, tmp_path, _ = station
    backup = create(cfg, tmp_path / "out")
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "x").write_text("mine")
    with pytest.raises(ValueError, match="not empty"):
        restore(backup, tmp_path / "busy")
    (backup / "files/models/a.pt").write_bytes(b"other")
    assert verify(backup) == ["changed: files/models/a.pt"]
    with pytest.raises(ValueError, match="not usable"):
        restore(backup, tmp_path / "new")


# ----------------------------------------------------------------------- health and alerts (E3, E12)

def test_healthcheck_fails_for_every_real_problem(tmp_path, monkeypatch):
    from src.config.config import Config
    from src.healthcheck import problems

    monkeypatch.delenv("CAMERA_STATUS_PATH", raising=False)
    status, db = tmp_path / "s.json", tmp_path / "e.db"
    cfg = Config(raw={"camera_manager": {"status_path": str(status)}, "logging": {"sqlite_path": str(db)}})
    assert "no runtime status" in problems(cfg)[0]
    sqlite3.connect(db).execute("CREATE TABLE t (a)").connection.commit()

    def write(**kw):
        status.write_text(json.dumps({"timestamp": time.time(), "workers": {"a": {}, "b": {}}, "problems": [], **kw}))
    write()
    assert problems(cfg) == []
    write(problems=["no frames from ['b']"])            # one dead camera of two is enough
    assert problems(cfg) == ["no frames from ['b']"]
    write(timestamp=time.time() - 60)
    assert "last reported" in problems(cfg)[0]
    write(workers={})
    assert problems(cfg) == ["no cameras are running"]
    write()
    lock = sqlite3.connect(db, isolation_level=None)
    db.unlink()
    assert "cannot be written" in problems(cfg)[0]
    lock.close()


def test_a_problem_is_sent_once_repeated_and_closed():
    from src.system.alerts import AlertSender

    current, sent = [set()], []
    sender = AlertSender({"webhook": "https://x", "hold_s": 10, "repeat_s": 100}, "line", lambda: list(current[0]))
    ok = [True]
    sender.send = lambda kind, text: (sent.append((kind, text)) or True) if ok[0] else False

    def at(t, *problems):
        current[0] = set(problems)
        sender.check(t)
    at(0, "disk")
    at(5, "disk")
    at(6)                       # gone before it mattered
    assert sent == []
    at(10, "cam")
    ok[0] = False
    at(20, "cam")               # delivery fails
    ok[0] = True
    at(25, "cam")
    at(30, "cam")
    assert sent == [("PROBLEM", "cam")]
    at(130, "cam")
    at(140)
    at(145)
    assert sent[1:] == [("PROBLEM", "cam"), ("OVER", "cam")]


def test_alert_channels_deliver_and_never_send_in_the_clear(tmp_path):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from src.system.alerts import AlertSender

    got = []

    class Hook(BaseHTTPRequestHandler):
        def do_POST(self):
            got.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a): pass
    server = HTTPServer(("127.0.0.1", 0), Hook)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    out = tmp_path / "andon"
    script = tmp_path / "andon.sh"
    script.write_text(f'#!/bin/sh\necho "$1|$2" >> {out}\n')
    script.chmod(0o755)
    sender = AlertSender({"webhook": f"http://127.0.0.1:{server.server_port}/", "command": [str(script)]}, "line", list)
    assert sender.send("PROBLEM", "camera; rm -rf /")
    server.shutdown()
    assert got[0]["text"] == "camera; rm -rf /" and out.read_text() == "PROBLEM|camera; rm -rf /\n"
    assert not AlertSender({"webhook": "http://example.com"}, "line", list).send("PROBLEM", "x")


def test_json_logs_are_one_object_per_line():
    from src.system.logs import JsonFormatter

    record = logging.LogRecord("x", logging.ERROR, "f", 1, "two\nlines %s", ("a",), None)
    assert json.loads(JsonFormatter().format(record))["message"] == "two\nlines a"
    assert "\n" not in JsonFormatter().format(record)


# ------------------------------------------------------------------------------------ MQTT (D7)

class _Broker:
    up, got = False, []

    def __init__(self, client_id):
        self.on_connect = self.on_disconnect = None

    def set_auth_credentials(self, *a): pass

    async def connect(self, *a, **k):
        if not _Broker.up:
            raise OSError("no broker")
        self.on_connect()

    def publish(self, topic, payload, qos):
        _Broker.got.append(json.loads(payload)["n"])

    async def disconnect(self): pass


def test_uplink_is_off_by_default_and_refuses_plain_connections(tmp_path):
    from src.network.mqtt_bridge import MqttBridge

    assert not MqttBridge({}).enabled and not MqttBridge(None).enabled
    with pytest.raises(ValueError, match="readable"):
        MqttBridge({"uplink_enabled": True, "use_tls": False, "spool_path": str(tmp_path / "s.db")})


def test_messages_survive_an_outage_and_a_restart_in_order(tmp_path):
    from src.network.mqtt_bridge import MqttBridge

    cfg = {"uplink_enabled": True, "use_tls": False, "allow_insecure": True, "spool_path": str(tmp_path / "s.db")}
    _Broker.up, _Broker.got = False, []

    async def first():
        bridge = MqttBridge(cfg, client_factory=_Broker)
        for n in range(3):
            await bridge.publish_telemetry("t", {"n": n})
        await asyncio.sleep(0.1)
        await bridge.close()

    async def second():
        bridge = MqttBridge(cfg, client_factory=_Broker)
        assert len(bridge.spool) == 3
        _Broker.up = True
        await bridge.start()
        await bridge.publish_telemetry("t", {"n": 3})
        await asyncio.sleep(0.6)
        await bridge.close()
    asyncio.run(first())
    asyncio.run(second())
    assert _Broker.got == [0, 1, 2, 3]


def test_spool_is_bounded(tmp_path):
    from src.network.mqtt_bridge import Spool

    spool = Spool(tmp_path / "s.db", max_items=5)
    for n in range(8):
        spool.put("t", str(n).encode(), 1)
    assert [p for _, _, p, _ in spool.oldest()] == [b"3", b"4", b"5", b"6", b"7"] and spool.dropped == 3


# -------------------------------------------------------------------------------- smaller things

def test_simulated_cycles_of_two_runs_do_not_collide():
    from src.process.simulator import InjectionMoldingSimulator
    from src.process.sources import SimulatedCycleSource

    ids = []
    for start in (1000.0, 2000.0):
        source = SimulatedCycleSource(InjectionMoldingSimulator(seed=1), time_scale=1000.0)
        source.subscribe(lambda c: (ids.append(c.cycle_id), source._stop.set()))
        import src.process.sources as module
        original, module.station_time = module.station_time, lambda s=start: s + 1e6
        try:
            source._run()
        finally:
            module.station_time = original
    assert len(set(ids)) == 2 and all(i.endswith("-1") for i in ids)


def test_camera_restart_waits_for_the_old_thread_and_drops_the_cut_off_part(tmp_path):
    from src.camera.part_inspection import PartPresenceTracker, PartVerdictFuser
    from src.process.synthetic_parts import render_part
    from test_model_ops import _StubStack, _reject_path_manager

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v", 0.4))
    cm.part_fuser = PartVerdictFuser()
    started, release = [], threading.Event()
    state.camera_id = "cam"
    state.capture_thread = threading.Thread(target=release.wait, daemon=True)
    state.capture_thread.start()
    cm.workers = {"cam": state}
    cm.monitor_interval = 0.02
    cm._camera_entries = lambda: [{"id": "cam"}]
    cm._publish_runtime_status = lambda: None
    tracker = PartPresenceTracker("cam", on_frames=1)
    tracker.update(render_part(160, 120, None, 1), True, 1.0, 1)
    cm.part_trackers = {"cam": tracker}
    decision = type("D", (), {"restart": True, "reason": "test"})
    cm.watchdog = type("W", (), {"evaluate": lambda s, st: decision, "record_recovered": lambda *a: None})()
    cm._start_camera_worker = lambda cfg, previous_restarts=0: (started.append(1), cm.shutdown_event.set())
    thread = threading.Thread(target=cm._monitor_loop)
    thread.start()
    time.sleep(0.8)
    assert not started                                  # the old reader still holds the camera
    release.set()
    thread.join(5)
    assert started and "cam" not in cm.part_trackers
    assert cm.fast_queue.get_nowait()["verdict"] == "not_inspected"


def test_full_disk_is_reported_not_ignored(tmp_path, monkeypatch):
    import src.inference.anomaly.monitoring as module

    queue = module.ReviewQueue(tmp_path / "r")
    monkeypatch.setattr(module.cv2, "imwrite", lambda *a: False)
    with pytest.raises(OSError):
        queue.add("c", np.zeros((8, 8, 3), np.uint8), normalized_score=1.0, verdict="pass")


def test_queues_do_not_list_their_folder_for_every_part(tmp_path, monkeypatch):
    from src.inference.good_pool import GoodPartPool

    pool = GoodPartPool(tmp_path, "a", every_n=1, max_items=3)
    frame = np.zeros((8, 8, 3), np.uint8)
    listed = []
    real = Path.glob
    monkeypatch.setattr(Path, "glob", lambda self, pattern: listed.append(pattern) or real(self, pattern))
    for _ in range(10):
        pool.offer(frame, verdict="pass", normalized_score=0.1, camera_id="c")
    assert listed.count("*.png") == 1 and len(pool) == 3


def test_evidence_images_stay_within_their_limit_oldest_first(tmp_path):
    from src.system.disk_monitor import DiskMonitor

    folder = tmp_path / "live" / "defects" / "cam"
    folder.mkdir(parents=True)
    for n in range(10):
        (folder / f"{n}.png").write_bytes(b"x" * 1000)
        os.utime(folder / f"{n}.png", (1000 + n, 1000 + n))
    (tmp_path / "live" / "keep.db").write_bytes(b"x" * 9000)
    DiskMonitor(watch_path=str(tmp_path), max_gb=6e-6)._keep_within_limit()
    assert sorted(f.name for f in folder.iterdir()) == ["5.png", "6.png", "7.png", "8.png", "9.png"]
    assert (tmp_path / "live" / "keep.db").exists()
