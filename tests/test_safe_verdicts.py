"""A plant must never get a wrong, stale or missing verdict: one test per way it used to happen."""
from __future__ import annotations

import json
import queue
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from src.camera.camera_manager import UNKNOWN_RECIPE, CameraManager
from src.camera.part_inspection import CompletedPart, PartPresenceTracker, PartVerdictFuser
from src.config.config import Config
from src.config.schema import ConfigError, check_real_plc, required_views
from src.inference.anomaly.monitoring import ReviewQueue
from src.process.synthetic_parts import render_part
from src.telemetry.plc_adapter import PLC_FLAG_NOT_INSPECTED, PLCAdapter, PLCAdapterConfig
from test_model_ops import _heartbeat_manager, _reject_path_manager, _StubStack


class Plc:
    """A Modbus TCP server with real registers; optionally acknowledges like a PLC program would."""

    def __init__(self, auto_ack: bool = False, sequence: int = 0):
        self.regs = {104: sequence, 105: sequence}
        self.auto_ack, self.writes, self.taken = auto_ack, [], []
        self.drop_reply_to = None
        self.server = socket.create_server(("127.0.0.1", 0))
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with conn:
            while True:
                try:
                    header = conn.recv(7)
                    if len(header) < 7:
                        return
                    body = conn.recv(struct.unpack(">H", header[4:6])[0] - 1)
                except OSError:
                    return
                function, address = body[0], struct.unpack(">H", body[1:3])[0]
                if function == 0x03:
                    reply = bytes([3, 2]) + struct.pack(">H", self.regs.get(address, 0))
                else:
                    values = ([struct.unpack(">H", body[3:5])[0]] if function == 0x06 else
                              list(struct.unpack(f">{body[5] // 2}H", body[6:6 + body[5]])))
                    for i, value in enumerate(values):
                        self.regs[address + i] = value
                    self.writes.append((address, values))
                    if address == 104:
                        self.taken.append((self.regs[100], self.regs[101], self.regs[104]))
                        if self.auto_ack:
                            self.regs[105] = self.regs[104]
                    if address == self.drop_reply_to:
                        self.drop_reply_to = None
                        return
                    reply = body[:5]
                conn.sendall(header[:4] + struct.pack(">H", len(reply) + 1) + header[6:7] + reply)

    def adapter(self, **kwargs) -> PLCAdapter:
        return PLCAdapter(PLCAdapterConfig(mode="plc", endpoint="127.0.0.1", port=self.server.getsockname()[1],
                                           **kwargs))


def _verdict(part: int, flag: int = 0) -> dict:
    return {"verdict": "fail" if flag else "pass", "flag": flag, "part": part}


# ------------------------------------------------------------------------- what the PLC sees
def test_sequence_changes_with_every_verdict_even_for_the_same_part():
    plc = Plc()
    adapter = plc.adapter(min_verdict_hold_s=0.0)
    for part in (4, 4, 0, 0):
        assert adapter.write_part_verdict(_verdict(part))
    assert [t[2] for t in plc.taken] == [1, 2, 3, 4]


def test_restart_continues_the_sequence_the_plc_last_saw_and_never_uses_zero():
    assert Plc(sequence=41).adapter().write_part_verdict(_verdict(1))
    plc = Plc(sequence=0xFFFF)
    assert plc.adapter().write_part_verdict(_verdict(1)) and plc.taken[0][2] == 1


def test_flag_and_part_are_in_place_before_the_sequence():
    plc = Plc()
    assert plc.adapter().write_part_verdict({**_verdict(12, 2), "cavity": 3})
    assert [address for address, _ in plc.writes] == [100, 106, 104] and plc.regs[106] == 3


def test_verdict_stays_long_enough_to_be_scanned():
    plc = Plc()
    adapter = plc.adapter(min_verdict_hold_s=0.15)
    adapter.write_part_verdict(_verdict(1))
    start = time.monotonic()
    adapter.write_part_verdict(_verdict(2))
    assert time.monotonic() - start >= 0.14


def test_verdict_is_not_overwritten_before_the_plc_acknowledged_it():
    plc = Plc()
    adapter = plc.adapter(require_ack=True, min_verdict_hold_s=0.0)
    assert adapter.write_part_verdict(_verdict(1, 3))
    assert not adapter.write_part_verdict(_verdict(2), time.monotonic() + 0.2)
    assert plc.regs[100] == 3 and plc.regs[101] == 1
    plc.regs[105] = plc.regs[104]
    assert adapter.write_part_verdict(_verdict(3), time.monotonic() + 1.0) and plc.regs[101] == 3


def test_acknowledging_plc_takes_every_verdict_in_order():
    plc = Plc(auto_ack=True)
    adapter = plc.adapter(require_ack=True, min_verdict_hold_s=0.0)
    assert all(adapter.write_part_verdict(_verdict(i)) for i in range(1, 30))
    assert [t[1] for t in plc.taken] == list(range(1, 30))


def test_lost_reply_does_not_deadlock_the_handshake():
    plc = Plc(auto_ack=True)
    adapter = plc.adapter(require_ack=True, min_verdict_hold_s=0.0)
    assert adapter.write_part_verdict(_verdict(1))
    plc.drop_reply_to = 104
    adapter.write_part_verdict(_verdict(2))          # delivered, but the reply never came
    assert adapter.write_part_verdict(_verdict(3), time.monotonic() + 2.0)
    assert plc.taken[-1][1] == 3


def test_restart_respects_an_unacknowledged_verdict():
    plc = Plc()
    plc.regs.update({100: 3, 101: 9, 104: 5, 105: 4})
    assert not plc.adapter(require_ack=True).write_part_verdict(_verdict(10), time.monotonic() + 0.2)
    assert plc.regs[100] == 3


def test_old_verdict_is_never_written_and_stops_the_heartbeat(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    wrote = []

    class Adapter:
        def write_part_verdict(self, signal, deadline):
            wrote.append(signal)
            return True

    cm._deliver_verdict(Adapter(), {**_verdict(1), "published_mono": time.monotonic() - 60})
    assert not wrote and cm._heartbeat_hold(time.monotonic()) == "a part verdict did not reach the PLC"
    records = [json.loads(x) for f in (tmp_path / "audit").glob("*.jsonl") for x in f.read_text().splitlines()]
    assert [r["record"] for r in records] == ["verdict_not_delivered"]


def test_unreachable_plc_gives_up_by_the_deadline_and_backlog_drains(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    cm.max_verdict_age_s, cm.audit_log = 0.3, None
    adapter = PLCAdapter(PLCAdapterConfig(mode="plc", endpoint="127.0.0.1", port=1))
    start = time.monotonic()
    for i in range(200):
        cm._deliver_verdict(adapter, {**_verdict(i), "published_mono": start})
    assert time.monotonic() - start < 2.0


def test_unmatched_part_gets_no_invented_number(tmp_path):
    cm, state = _heartbeat_manager(tmp_path, ["cam"])
    cm.part_fuser = PartVerdictFuser()
    state.camera_id = "cam"
    for key in ("m-3", None, "m-4#2"):
        cm._report_view(state, key, 0)
    sent = [cm.fast_queue.get_nowait() for _ in range(3)]
    assert [(s["part"], s["cavity"]) for s in sent] == [(3, None), (0, None), (4, 2)]


# ------------------------------------------------------------------------------- one part, many cameras
def test_second_camera_cannot_turn_a_reject_into_a_pass():
    fuser = PartVerdictFuser()
    assert fuser.add("m-1", "a", 3, 0.0)["verdict"] == "fail"
    assert fuser.add("m-1", "b", 0, 0.1) is None
    assert fuser.add("m-2", "a", 0, 1.0)["verdict"] == "pass"
    late = fuser.add("m-2", "b", 2, 1.1)
    assert late["verdict"] == "fail" and late["update"]


def test_unmarked_multi_camera_station_requires_every_camera():
    cams = [{"id": "a"}, {"id": "b"}, {"id": "off", "enabled": False}, {"id": "c", "required_view": False}]
    assert required_views(cams) == ["a", "b"]
    assert required_views([{"id": "a", "required_view": True}, {"id": "b"}]) == ["a"]
    assert required_views([{"id": "only"}]) == []


def test_optional_camera_can_reject_but_never_pass():
    fuser = PartVerdictFuser(["a"], timeout_s=5.0)
    assert fuser.add("m-1", "extra", 0, 0.0) is None
    assert fuser.add("m-1", "extra", 2, 0.1) is None      # kept until the part is decided
    assert fuser.add("m-1", "a", 0, 0.2)["verdict"] == "fail"
    assert fuser.add("m-2", "a", 0, 1.0)["verdict"] == "pass"
    assert fuser.add("m-2", "extra", 1, 1.5)["verdict"] == "fail"
    assert fuser.add(None, "extra", 3, 2.0) is None
    assert fuser.expired(8.0)[0]["verdict"] == "fail"


def test_reject_after_the_timeout_still_reaches_the_plc_once():
    fuser = PartVerdictFuser(["a", "b", "c"], timeout_s=5.0)
    fuser.add("m-1", "a", 0, 0.0)
    assert fuser.expired(6.0)[0]["verdict"] == "not_inspected"
    late = fuser.add("m-1", "b", 3, 7.0)
    assert late["update"] and late["flag"] == 3 and late["cycle_id"] == "m-1" and late["missing"] == ["c"]
    assert fuser.add("m-1", "c", 0, 8.0) is None           # cannot soften it
    assert [v["flag"] for v in fuser.unseen(["m-2"])] == [PLC_FLAG_NOT_INSPECTED]
    assert fuser.add("m-2", "a", 2, 9.0)["verdict"] == "fail"


def test_late_reject_does_not_hold_the_heartbeat_of_a_healthy_station(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    cm._publish_part_verdict({"cycle_id": "m-1", "verdict": "fail", "flag": 2, "views": {}, "missing": ["x"],
                              "update": True})
    assert cm._heartbeat_hold(time.monotonic()) is None


def test_blind_optional_camera_does_not_stop_the_line(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    cm._view_faults = {"optional": True}
    assert cm._heartbeat_hold(time.monotonic()) is None
    cm._view_faults = {"top": True}
    assert cm._heartbeat_hold(time.monotonic())


# ----------------------------------------------------------------------------------- shutdown
def test_every_part_seen_before_shutdown_gets_a_verdict_at_the_plc(tmp_path):
    cm, state = _heartbeat_manager(tmp_path, ["cam"])
    cm.part_fuser = PartVerdictFuser()
    cm.main_loop, cm.alert_queue = None, queue.Queue()
    cm.inspection_enabled = True
    cm.inspection_executor = ThreadPoolExecutor(1)
    cm.executor, cm.vlm_executor = ThreadPoolExecutor(1), ThreadPoolExecutor(1)
    cm._monitor_thread = cm._disk_writer_thread = None
    cm._stopped = False
    cm.workers = {"cam": state}
    state.camera_id, state.capture_thread, state.inference_thread = "cam", None, None
    idle = type("Idle", (), {"stop": lambda self: None})
    cm.multiplexer, cm.runtime_metrics, cm.disk_monitor = idle(), idle(), idle()

    tracker = PartPresenceTracker("cam", on_frames=1)
    tracker.update(render_part(160, 120, None, seed=1), True, 1.0, 1)     # a part is in view right now
    cm.part_trackers = {"cam": tracker}
    written = []

    class Adapter:
        is_simulated = True

        def write_part_verdict(self, signal, deadline):
            written.append(signal["verdict"])
            return True

        def write_heartbeat(self):
            return True

    cm.telemetry = type("T", (), {"adapter": Adapter()})()
    cm._reflex_thread = threading.Thread(target=cm._reflex_worker, daemon=True)
    cm._reflex_thread.start()
    frame = render_part(160, 120, None, seed=3)
    slow = cm._inspect_part
    cm._inspect_part = lambda s, p: (time.sleep(0.1), slow(s, p))
    for i in range(4):
        cm._inspection_slots.acquire()
        cm.inspection_executor.submit(cm._inspect_queued, state, CompletedPart("cam", 1.0, 2.0, frame, i, 3))
    cm._sync_stop()
    assert written == ["pass"] * 4 + ["not_inspected"]
    assert not cm._reflex_thread.is_alive()


def test_heartbeat_stops_as_soon_as_shutdown_begins(tmp_path):
    cm, _ = _heartbeat_manager(tmp_path)
    cm.shutdown_event.set()
    assert cm._heartbeat_hold(time.monotonic()) == "inspection is shutting down"


# --------------------------------------------------------------------- fake images, wrong models
def test_simulated_images_cannot_drive_a_real_plc():
    plc = {"telemetry": {"mode": "plc"}}
    real = [{"id": "a", "source": "rtsp://x"}]
    check_real_plc({**plc, "cameras": real})
    for bad in ([{"id": "a", "source": "synthetic://a"}], [{"id": "a", "source": 0, "mock_inference": True}]):
        with pytest.raises(ConfigError):
            check_real_plc({**plc, "cameras": bad})
        check_real_plc({"telemetry": {"mode": "simulated"}, "cameras": bad})
        check_real_plc({"telemetry": {"mode": "plc", "bench_test": True}, "cameras": bad})
    with pytest.raises(ConfigError):
        check_real_plc({**plc, "cameras": real}, forced_by_env=True)


def test_env_flag_cannot_replace_real_cameras_on_a_real_plc(monkeypatch):
    monkeypatch.setenv("QUALYXUS_USE_SYNTHETIC_CAMERAS", "1")
    with pytest.raises(ConfigError):
        CameraManager(config_data={"telemetry": {"mode": "plc"}, "cameras": [{"id": "a", "source": "rtsp://x"}]})


def test_dead_camera_is_never_replaced_by_a_simulated_one():
    import inspect

    from src.camera import camera_manager

    assert "fallback_to_synthetic" not in inspect.getsource(camera_manager)


def test_unknown_product_is_not_judged_by_another_products_model(tmp_path):
    from src.inference.inspection_stack import StackProvider

    provider = StackProvider({"anomaly_model": str(tmp_path / "default.pt"), "registry_dir": str(tmp_path / "reg"),
                              "recipes": {"known": {"anomaly_model": str(tmp_path / "known.pt")}}})
    stack, shadow = provider.get("SOMETHING_ELSE")
    assert stack.empty and not stack.model_missing and shadow is None
    assert provider.get("known")[0].model_missing and provider.get()[0].model_missing   # these do look for a file


def test_machine_that_does_not_name_the_product_gets_no_model(tmp_path):
    cm, _ = _reject_path_manager(tmp_path, _StubStack(False, "v", 0.4))
    cm.recipe_from_scalar = "mold_id"
    assert cm._recipe_for("cam", "m-1") == UNKNOWN_RECIPE and cm._recipe_for("cam", None) == UNKNOWN_RECIPE


# --------------------------------------------------------------------------------- retraining data
def test_calibration_sample_keeps_hard_parts_and_waits_for_a_person(tmp_path):
    from src.inference.good_pool import GoodPartPool

    pool = GoodPartPool(tmp_path, "housing", calibration_every_n=1)
    image = np.zeros((8, 8, 3), np.uint8)
    cases = [("pass", 0.3, "a"), ("pass", 0.95, "b"), ("fail", 1.4, "c"), ("fail", 2.0, "d")]
    got = [pool.sample(image, verdict=v, normalized_score=s, camera_id="cam", image_sha=sha) for v, s, sha in cases]
    assert got == ["accepted", "needs_review", "needs_review", "needs_review"]
    images, waiting = pool.calibration_set({})
    assert (len(images), waiting) == (1, 3)
    images, waiting = pool.calibration_set({"b": "pass", "c": "pass", "d": "fail"})   # c was a false reject
    assert (len(images), waiting) == (3, 0)
    assert len(pool) == 0        # calibration parts are never trained on


def test_unbiased_calibration_keeps_the_false_reject_promise():
    from src.inference.anomaly.calibration import conformal_threshold

    rng = np.random.default_rng(0)
    old = conformal_threshold(rng.lognormal(0, 0.25, 5000), 0.01)["threshold"]
    good = rng.lognormal(0, 0.25, 20000)
    fresh = rng.lognormal(0, 0.25, 50000)
    truncated = conformal_threshold(good[good <= 0.8 * old][:1500], 0.01)["threshold"]
    unbiased = conformal_threshold(good[:1500], 0.01)["threshold"]
    assert (fresh > truncated).mean() > 0.05       # what retraining used to do
    assert (fresh > unbiased).mean() < 0.016


def test_uncertain_calibration_part_is_sent_to_the_review_queue(tmp_path):
    cm, state = _reject_path_manager(tmp_path, _StubStack(True, "v0001", 1.9))    # outside the review band
    cm.review_queue = ReviewQueue(tmp_path / "review")
    cm._pool_cfg["calibration_every_n"] = 1
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    items = [json.loads(p.read_text()) for p in (tmp_path / "review").rglob("*.json")]
    assert len(items) == 1 and items[0]["image_sha"]


def test_full_disk_for_samples_does_not_cost_the_audit_record(tmp_path, monkeypatch):
    from src.inference import good_pool

    cm, state = _reject_path_manager(tmp_path, _StubStack(False, "v0001", 0.4))
    cm._pool_cfg["calibration_every_n"] = 1
    monkeypatch.setattr(good_pool.cv2, "imwrite", lambda *a: False)
    cm.review_queue = None
    cm._inspect_part(state, CompletedPart("cam", 1.0, 2.0, render_part(160, 120, None, seed=3), 7, 3))
    records = [json.loads(x) for f in (tmp_path / "audit").glob("*.jsonl") for x in f.read_text().splitlines()]
    assert len(records) == 2 and state.parts_inspected == 1
    assert not list((tmp_path / "pool").rglob("*.json"))


# ------------------------------------------------------------------------------------------ config
@pytest.mark.parametrize("text,message", [
    ("cameras:\n - {id: a, requred_view: true}\n", "did you mean 'required_view'"),
    ("inspecton: {enabled: true}\n", "did you mean 'inspection'"),
    ("inspection: {view_timout_s: 1}\n", "did you mean 'view_timeout_s'"),
    ("telemetry: {register_map: {verdct: 5}}\n", "did you mean 'verdict'"),
    ("cameras:\n - {id: a, process_link: {tolerence_s: 1}}\n", "did you mean 'tolerance_s'"),
    ("cameras:\n - {id: a, process_link: 3}\n", "must be a section"),
    ("cameras:\n - {id: a}\n - {id: a}\n", "used more than once"),
    ("cameras:\n - {source: x}\n", "needs an id"),
    ("cameras:\n - {id: a, required_view: 'false'}\n", "must be true or false"),
    ("cameras:\n - {id: a, required_view: false}\n - {id: b, required_view: false}\n", "no camera is required"),
    ("inspection: {view_timeout_s: 0}\n", "positive number"),
    ("telemetry: {mode: PLC}\n", "telemetry.mode"),
    ("telemetry: {register_map: {verdict: 102}}\n", "one register for two values"),
    ("inspection: {on_frames: 1}\ninspection: {off_frames: 1}\n", "appears twice"),
    ("cameras: {id: a}\n", "must be a list"),
])
def test_config_mistakes_stop_the_start(tmp_path, monkeypatch, text, message):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c.yaml").write_text(text)
    with pytest.raises(ConfigError, match=message):
        Config.load(config_path=str(tmp_path / "c.yaml"))


def test_config_layers_go_to_any_depth_and_never_loop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.yaml").write_text("inspection: {on_frames: 1, off_frames: 1}\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\ninspection: {on_frames: 2}\n")
    (tmp_path / "c.yaml").write_text("extends: b.yaml\ninspection: {recipe: x}\n")
    raw = Config.load(config_path=str(tmp_path / "c.yaml")).raw
    assert raw["inspection"] == {"on_frames": 2, "off_frames": 1, "recipe": "x"}
    (tmp_path / "a.yaml").write_text("extends: c.yaml\npaths: {}\n")
    with pytest.raises(ConfigError, match="circle"):
        Config.load(config_path=str(tmp_path / "c.yaml"))


@pytest.mark.parametrize("name", ["config.yaml", "config.360.yaml", "src/config/default.yaml"])
def test_shipped_configs_are_valid(name, tmp_path, monkeypatch):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / name
    monkeypatch.chdir(tmp_path)
    Config.load(config_path=str(path))
