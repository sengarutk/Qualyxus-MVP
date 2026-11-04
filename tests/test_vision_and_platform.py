from __future__ import annotations

import queue

import base64
import os

import numpy as np
import pytest

from src.camera.part_inspection import PartPresenceTracker
from src.process.synthetic_parts import BACKGROUND, render_part


def _blank(h=120, w=160):
    return np.full((h, w, 3), BACKGROUND, dtype=np.uint8)


def test_part_tracker_yields_one_record_per_visit():
    tracker = PartPresenceTracker("cam", on_frames=2, off_frames=2)
    part = render_part(160, 120, None, seed=1)
    sequence = [False] * 3 + [True] * 6 + [False] * 4 + [True] * 5 + [False] * 3
    done = []
    for i, present in enumerate(sequence):
        frame = part if present else _blank()
        result = tracker.update(frame, present, float(i), i, ["e1"] if i == 5 else [], ["anomaly"] if i == 5 else [])
        if result:
            done.append(result)
    assert len(done) == 2
    assert done[0].defect_event_ids == ["e1"] and not done[1].defect_event_ids
    assert done[0].first_ts <= done[0].ts <= done[0].last_ts


def test_part_tracker_debounces_single_frame_glitches():
    tracker = PartPresenceTracker("cam", on_frames=2, off_frames=2)
    out = [tracker.update(_blank(), p, float(i), i) for i, p in enumerate([False, True, False, False, True, False])]
    assert all(r is None for r in out)


@pytest.fixture(scope="module")
def detector():
    from src.inference.anomaly.patchcore import PatchCoreDetector

    det = PatchCoreDetector(backbone="resnet18", image_size=160)
    det.fit([render_part(320, 240, None, seed=100 + i) for i in range(30)])
    return det


@pytest.mark.parametrize("defect", ["short_shot", "flash", "sink_mark", "burn_mark"])
def test_patchcore_detects_defects_trained_on_good_parts_only(detector, defect):
    hits = [detector.predict(render_part(320, 240, defect, seed=900 + i)).is_anomalous for i in range(5)]
    assert sum(hits) >= 4


def test_patchcore_accepts_unseen_good_parts_and_localises(detector):
    good = [detector.predict(render_part(320, 240, None, seed=700 + i)) for i in range(8)]
    assert sum(r.is_anomalous for r in good) <= 1
    bad = detector.predict(render_part(320, 240, "short_shot", seed=3))
    assert bad.bbox is not None and bad.anomaly_map.shape == (240, 320)


def test_patchcore_save_load_roundtrip(tmp_path, detector):
    from src.inference.anomaly.patchcore import PatchCoreDetector

    path = tmp_path / "pc.pt"
    detector.save(path)
    loaded = PatchCoreDetector.load(path)
    img = render_part(320, 240, "flash", seed=5)
    assert loaded.predict(img).score == pytest.approx(detector.predict(img).score, rel=1e-2)


def test_federated_payload_is_authenticated(tmp_path, monkeypatch):
    import torch

    from src.config.config import Config
    from src.training.federated_client import FederatedClient
    from src.training.federated_crypto import FederatedKeyError, generate_key
    from src.training.federated_server import FederatedServer

    monkeypatch.setenv("QUALYXUS_FEDERATED_KEY", generate_key())
    model = torch.nn.Module()
    model.head = torch.nn.Linear(3, 2)
    client = FederatedClient(Config(raw={"paths": {"data_dir": str(tmp_path)}}), node_id="plant")
    payload = client.export_secure_payload(model, "v1")
    assert "head.weight" in FederatedServer().load_secure_payload(payload)
    with pytest.raises(ValueError):
        FederatedServer().load_secure_payload(dict(payload, node_id="evil"))
    with pytest.raises(ValueError):
        FederatedServer(key=base64.urlsafe_b64decode(generate_key())).load_secure_payload(payload)
    monkeypatch.delenv("QUALYXUS_FEDERATED_KEY")
    with pytest.raises(FederatedKeyError):
        FederatedServer()


def test_config_dataset_yaml_paths_agree_and_zero_holds_are_honoured(tmp_path):
    from src.config.config import Config

    cfg = Config(raw={"inference": {"yolo_yaml": str(tmp_path / "r.yaml")},
                      "resource_management": {"critical_hold_seconds": 0, "warning_hold_seconds": 0}})
    assert cfg.get("inference", "yolo_yaml") == cfg.get("dataset", "yolo", "yaml_path")
    assert cfg.resource_management.critical_hold_seconds == 0.0


def test_env_override_keeps_loaded_config(monkeypatch, tmp_path):
    from src.config.config import Config, env_override_config

    base = Config(raw={"marker": 1, "dataset": {"name": "a"}})
    monkeypatch.setenv("DATASET_NAME", "b")
    monkeypatch.setenv("SEVERSTAL_TRAIN_CSV", str(tmp_path / "train.csv"))
    out = env_override_config(base)
    assert out.raw["marker"] == 1 and out.raw["dataset"]["name"] == "b"
    assert out.get("severstal", "train_csv") == str(tmp_path / "train.csv")


def test_uncalibrated_vibration_never_classifies_or_flags(tmp_path):
    from src.inference.analysis.vibration_engine import VibrationAnalyzer

    vib = VibrationAnalyzer(buffer_size=64, kap_db_path=str(tmp_path / "kap.db"), camera_id="c", calibrated=False)
    vib._fast_queue = queue.Queue()
    ts = 0.0
    result = None
    for i in range(64):
        ts += 1 / 30.0
        result = vib.ingest(50.0 * np.sin(i), 0.0, ts=ts)
    assert result["iso_status"] == "UNCALIBRATED" and vib._fast_queue.empty()
