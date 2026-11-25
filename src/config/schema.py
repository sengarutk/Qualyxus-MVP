"""Config checks that run at load time: a typo must never silently remove a safety rule.

Sections that decide verdicts (cameras, inspection, telemetry, process) accept only the keys
the software reads. ``ANY`` marks a section or value that is not checked further. Keys
starting with ``x-`` are free (YAML anchors).
"""
from __future__ import annotations

import difflib
from typing import Any, Iterable

ANY = object()


class ConfigError(ValueError):
    pass


def _keys(names: str) -> dict[str, Any]:
    return {name: ANY for name in names.split()}


_SIGNAL = _keys("kind scale offset address register_type data_type word_order node_id")
_LINK = _keys("machine_id mode transport_delay_s tolerance_s fifo_offset parts_per_cycle")
_MODELS = _keys("anomaly_model segmentation_model defect_type_model")
_CAMERA = {
    **_keys("""id enabled source mock_inference mock_event_every independent_logs fps width height frame_queue_size
               event_queue_size jpeg_quality reconnect_delay save_every buffer_size raw_dir defect_dir recipe
               required_view synthetic_scene synthetic_fail_after_frames part_dwell_s view_index n_views
               inject_defect_every model_view"""),
    "presence": _keys("min_contrast min_area reference"),
    "process_link": _LINK,
}
_CONNECTION = {**_keys("""host port unit_id poll_hz endpoint username password password_env security_policy
                          security_mode certificate private_key server_certificate allow_insecure_login"""),
               "cycle_counter": _SIGNAL, "signals": {"*": _SIGNAL}, "setpoints": {"*": _SIGNAL}}

SCHEMA: dict[str, Any] = {
    **_keys("""paths active_profile dataset inference training logging vlm_reasoning
               resource_management camera_manager
               watchdog disk_monitor runtime_metrics hardening test_flags camera storage
               vibration_analysis geometry severstal severity"""),
    "telemetry": {**_keys("""mode protocol endpoint port unit_id heartbeat_s require_ack min_verdict_hold_s
                             max_verdict_age_s bench_test"""),
                  "register_map": _keys("verdict part heartbeat alert sequence ack cavity trigger")},
    "inspection": {
        **_keys("""enabled recipe recipe_from_scalar registry_dir reload_check_s max_cached_recipes max_frame_age_s
                   max_inspect_s max_inspection_backlog expect_part_per_cycle view_timeout_s
                   on_frames off_frames"""),
        **_MODELS,
        "recipes": {"*": _MODELS},
        "shadow": _keys("enabled"),
        "presence": _keys("min_contrast min_area reference"),
        "trigger": _keys("enabled frames delay_s poll_s"),
        "health": _keys("window drift_ratio expected_reject_rate request_retrain"),
        "audit": _keys("enabled dir key_file"),
        "good_pool": _keys("enabled dir every_n max_items max_score calibration_every_n calibration_max_items max_gb"),
        "review": _keys("enabled dir band max_gb"),
        "view_store": _keys("enabled dir keep_parts"),
    },
    "process": {
        **_keys("""enabled source machine_id line_id db_path models_dir store_curves curve_retention_days spec_limits
                   components baseline_reset_on view_spread_s retention_days"""),
        "simulator": _keys("seed time_scale disturbance_scale scenario"),
        "model": _keys("""golden_warmup_cycles retrain_every_labels min_train_samples min_train_positives
                          min_promotion_auc min_label_confidence regime_min_good_cycles hybrid_deviation
                          max_train_cycles"""),
        "prescriptions": _keys("enabled limits knowledge max_step_fraction min_risk_drop cooldown_cycles"),
        "modbus": _CONNECTION,
        "opcua": _CONNECTION,
        "machines": [{**_keys("machine_id line_id source"), "simulator": _keys("seed time_scale disturbance_scale scenario"),
                      "modbus": _CONNECTION, "opcua": _CONNECTION}],
    },
    "cameras": [_CAMERA],
    "api": _keys("host port tls_cert tls_key behind_proxy"),
    "mqtt": _keys("""uplink_enabled host port use_tls allow_insecure ca_certs certfile keyfile username password_env
                     client_id spool_path"""),
    "console": _keys("status_path users_file"),
    "alerts": {**_keys("webhook webhook_token_env command hold_s repeat_s"),
               "email": _keys("host port starttls username password_env from to")},
}

# Values whose wrong type would change a verdict rule ("false" in quotes is true in Python).
_BOOLEANS = {"enabled", "required_view", "mock_inference", "require_ack", "expect_part_per_cycle", "bench_test",
             "request_retrain", "store_curves"}
_POSITIVE = {"view_timeout_s", "max_verdict_age_s", "max_inspect_s", "max_frame_age_s", "max_inspection_backlog",
             "heartbeat_s", "tolerance_s", "fps", "width", "height", "on_frames", "off_frames", "min_contrast", "min_area", "poll_s", "max_gb", "view_spread_s"}
_NOT_NEGATIVE = {"min_verdict_hold_s", "transport_delay_s", "delay_s"}


def _check(value: Any, schema: Any, path: str, problems: list[str]) -> None:
    if isinstance(schema, list):
        if value is None:
            return
        if not isinstance(value, list):
            problems.append(f"{path} must be a list")
            return
        for i, item in enumerate(value):
            _check(item, schema[0], f"{path}[{i}]", problems)
        return
    if schema is ANY or value is None:
        return
    if not isinstance(value, dict):
        problems.append(f"{path} must be a section, got {value!r}")
        return
    for key, item in value.items():
        where = f"{path}.{key}" if path else str(key)
        if str(key).startswith("x-"):
            continue
        rule = schema.get(key, schema.get("*"))
        if rule is None:
            close = difflib.get_close_matches(str(key), [k for k in schema if k != "*"], n=1)
            problems.append(f"unknown key '{where}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
            continue
        _check_value(key, item, where, problems)
        _check(item, rule, where, problems)


def _check_value(key: str, value: Any, where: str, problems: list[str]) -> None:
    if isinstance(value, (dict, list)) or value is None:
        return
    if key in _BOOLEANS:
        if not isinstance(value, bool):
            problems.append(f"{where} must be true or false, got {value!r}")
    elif key in _POSITIVE or key in _NOT_NEGATIVE:
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if not number or value < 0 or (key in _POSITIVE and value == 0):
            problems.append(f"{where} must be a {'positive ' if key in _POSITIVE else ''}number, got {value!r}")


def validate(raw: dict[str, Any]) -> None:
    """Raise ConfigError listing every problem found."""
    problems: list[str] = []
    _check(raw, SCHEMA, "", problems)
    cameras = [c for c in raw.get("cameras") or [] if isinstance(c, dict)]
    ids = [str(c["id"]) if c.get("id") is not None else None for c in cameras]
    if None in ids:
        problems.append("every camera needs an id")
    for duplicate in sorted({i for i in ids if i is not None and ids.count(i) > 1}):
        problems.append(f"camera id '{duplicate}' is used more than once")
    enabled = [c for c in cameras if c.get("enabled", True)]
    if len(enabled) > 1 and not required_views(enabled):
        problems.append("no camera is required: with several cameras at least one needs required_view: true")
    process = raw.get("process") if isinstance(raw.get("process"), dict) else {}
    machines = [m for m in process.get("machines") or [] if isinstance(m, dict)]
    names = [m.get("machine_id") for m in machines]
    if None in names or len(set(names)) != len(names):
        problems.append("every process.machines entry needs its own machine_id")
    telemetry = raw.get("telemetry") if isinstance(raw.get("telemetry"), dict) else {}
    if telemetry.get("mode") not in (None, "simulated", "plc"):
        problems.append(f"telemetry.mode must be 'simulated' or 'plc', got {telemetry.get('mode')!r}")
    inspection = raw.get("inspection") if isinstance(raw.get("inspection"), dict) else {}
    if (inspection.get("trigger") or {}).get("enabled") and telemetry.get("mode") != "plc":
        problems.append("inspection.trigger needs telemetry.mode: plc (the trigger comes from the PLC)")
    registers = telemetry.get("register_map")
    if isinstance(registers, dict):
        from src.telemetry.plc_adapter import PLC_REGISTERS

        used = list({**PLC_REGISTERS, **registers}.values())
        if any(not isinstance(v, int) or isinstance(v, bool) for v in used):
            problems.append("telemetry.register_map values must be register numbers")
        elif len(set(used)) != len(used):
            problems.append("telemetry.register_map uses one register for two values")
    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def required_views(cameras: Iterable[dict[str, Any]]) -> list[str]:
    """Cameras that must all pass a part before the part passes.

    The cameras marked ``required_view: true``. With none marked, every camera of a
    multi-camera station is required unless it says ``required_view: false``, so one
    camera's pass can never stand in for another's. A lone camera needs no marking.
    """
    enabled = [c for c in cameras if c.get("enabled", True)]
    marked = [str(c.get("id")) for c in enabled if c.get("required_view")]
    if marked or len(enabled) < 2:
        return marked
    return [str(c.get("id")) for c in enabled if c.get("required_view") is None]


def check_real_plc(raw: dict[str, Any], forced_by_env: bool = False) -> None:
    """Simulated images or mock inference must never drive a real PLC by accident."""
    telemetry = raw.get("telemetry") or {}
    if str(telemetry.get("mode") or "simulated") == "simulated" or telemetry.get("bench_test"):
        return
    fake = [str(c.get("id")) for c in raw.get("cameras") or [] if c.get("enabled", True) and (
        forced_by_env or c.get("mock_inference") or str(c.get("source", "")).startswith("synthetic://"))]
    if fake:
        raise ConfigError(f"Simulated images or mock inference on {fake} cannot drive a real PLC. For a bench test "
                          "with no production parts, set telemetry.bench_test: true")
