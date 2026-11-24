"""Everything the operations console shows, read from what the runtime actually recorded.

Nothing here is written back and nothing is estimated silently: a value with no data behind
it is returned as None, and the console shows a dash.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.camera.view_store import part_folder
from src.process.machine_health import ACT_Z, INJECTION_MOLDING_COMPONENTS, WATCH_Z, assess_machine
from src.process.simulator import INJECTION_MOLDING_SPEC_LIMITS
from src.process.store import ProcessStore

BASELINE_CYCLES = 100
STATUS_STALE_S = 20.0

# feature -> (label, unit, setpoint it should follow)
SIGNALS: dict[str, tuple[str, str, str | None]] = {
    "hold_pressure.active_median": ("Hold pressure", "bar", "setpoint.hold_pressure_sp"),
    "cushion_mm": ("Cushion", "mm", None),
    "melt_temp.active_median": ("Melt temperature", "°C", "setpoint.melt_temp_sp"),
    "mold_temp.active_median": ("Mould temperature", "°C", "setpoint.mold_temp_sp"),
    "injection_pressure.max": ("Peak injection pressure", "bar", None),
    "cavity_pressure.max": ("Peak cavity pressure", "bar", None),
    "dryer_dewpoint_c": ("Dryer dew point", "°C", None),
}
CURVES = {"injection_pressure": ("Injection pressure", "bar"), "cavity_pressure": ("Cavity pressure", "bar"),
          "hold_pressure": ("Hold pressure", "bar"), "melt_temp": ("Melt temperature", "°C"),
          "mold_temp": ("Mould temperature", "°C")}
# What each signal leaving its baseline is consistent with. A pointer for maintenance, not proof.
POINTS_TO = {
    "hold_pressure.active_median": (-1, "Check ring"), "cushion_mm": (-1, "Check ring"),
    "melt_temp.active_median": (-1, "Heater bands"), "mold_temp.active_median": (1, "Mould cooling"),
    "dryer_dewpoint_c": (1, "Material dryer"),
}
CURVE_POINTS = 160


def _robust(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    return center, max(1.4826 * float(np.median(np.abs(values - center))), 1e-6)


def _num(value: Any) -> float | None:
    return float(value) if value is not None and np.isfinite(value) else None


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


class ConsoleData:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        process = cfg.get("process") or {}
        inspection = cfg.get("inspection") or {}
        self.db_path = process.get("db_path", "data/events/process.db")
        self.actions_db = str(Path(self.db_path).parent / "actions.db")
        self.components = process.get("components") or INJECTION_MOLDING_COMPONENTS
        self.status_path = (cfg.get("console") or {}).get("status_path", "data/live/camera_status.json")
        self.views_dir = Path((inspection.get("view_store") or {}).get("dir", "data/live/parts"))
        audit = inspection.get("audit") or {}
        self.audit_dir, self.audit_key = audit.get("dir", "data/audit"), audit.get("key_file")
        self.model_path = inspection.get("anomaly_model")
        cameras = [c for c in cfg.get("cameras") or [] if c.get("enabled", True)]
        self.cameras = [str(c["id"]) for c in cameras]
        self.required = [str(c["id"]) for c in cameras if c.get("required_view")] or self.cameras
        self.simulated = str(process.get("source", "simulated")) == "simulated"
        self.time_scale = float((process.get("simulator") or {}).get("time_scale", 1.0)) if self.simulated else 1.0
        self._store: ProcessStore | None = None
        self._lock = threading.RLock()
        self._audit_cache: tuple[float, dict[str, Any]] = (0.0, {})
        self._model_cache: dict[str, Any] | None = None

    @property
    def store(self) -> ProcessStore | None:
        with self._lock:
            if self._store is None and Path(self.db_path).exists():
                self._store = ProcessStore(self.db_path)
            return self._store

    def _cycles(self, limit: int, newest: bool = True) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        order = "DESC" if newest else "ASC"
        rows = self.store._query(
            f"SELECT cycle_id, sequence, end_ts, features FROM process_cycles ORDER BY end_ts {order} LIMIT ?", (limit,))
        out = [{"cycle_id": r["cycle_id"], "sequence": r["sequence"], "ts": float(r["end_ts"]),
                "features": json.loads(r["features"])} for r in rows]
        return out[::-1] if newest else out

    # -------------------------------------------------------------------------------- parts

    def parts(self, window_s: float = 8 * 3600.0, now: float | None = None) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        now = time.time() if now is None else now
        parts = self.store.part_verdicts(now - window_s, now + 60.0, required_cameras=self.required)
        # A part still moving through the cameras is not a result yet.
        settle = float((self.cfg.get("inspection") or {}).get("view_timeout_s", 5.0))
        return [p for p in parts if not (p["verdict"] == "incomplete" and now - p["ts"] < settle + 3.0)]

    @staticmethod
    def _key(part: dict[str, Any]) -> str:
        return f"{part['cycle_id']}#{part['cavity'] or 0}" if part["cycle_id"] else ""

    def _images(self, cycle_id: str, cavity: int) -> dict[str, dict[str, str]]:
        for key in (cycle_id, f"{cycle_id}#{cavity}"):
            folder = self.views_dir / part_folder(key)
            if folder.is_dir():
                out: dict[str, dict[str, str]] = {}
                for camera in self.cameras:
                    name = part_folder(camera)
                    shots = {}
                    if (folder / f"{name}.jpg").exists():
                        shots["image"] = f"/img/{folder.name}/{name}.jpg"
                    if (folder / f"{name}.heat.jpg").exists():
                        shots["heat"] = f"/img/{folder.name}/{name}.heat.jpg"
                    out[camera] = shots
                return out
        return {}

    def part_list(self, limit: int = 120) -> list[dict[str, Any]]:
        parts = sorted(self.parts(), key=lambda p: p["ts"])[-limit:]
        return [{"key": self._key(p), "cycle_id": p["cycle_id"], "cavity": p["cavity"] or 0, "ts": p["ts"],
                 "verdict": p["verdict"], "defects": p["defect_types"], "missing": p["missing_views"],
                 "failed_views": [c for c, v in p["views"].items() if v == "fail"],
                 "number": int(str(p["cycle_id"]).rsplit("-", 1)[-1]) if p["cycle_id"]
                 and str(p["cycle_id"]).rsplit("-", 1)[-1].isdigit() else None} for p in parts]

    def part_detail(self, cycle_id: str, cavity: int = 0) -> dict[str, Any] | None:
        store = self.store
        cycle = store.get_cycle(cycle_id, with_curves=True) if store is not None else None
        if cycle is None:
            return None
        rows = store._query("SELECT camera_id, verdict, anomaly_score, defect_type, ts, alignment_confidence, source "
                            "FROM inspections WHERE cycle_id = ? AND cavity = ? ORDER BY ts", (cycle_id, cavity))
        images = self._images(cycle_id, cavity)
        by_camera = {r["camera_id"]: r for r in rows if r["source"] == "vision"}
        views = []
        for camera in self.cameras:
            r = by_camera.get(camera)
            views.append({"camera": camera, "verdict": r["verdict"] if r else "missing",
                          "score": _num(r["anomaly_score"]) if r else None, **images.get(camera, {})})
        confidences = [r["alignment_confidence"] for r in by_camera.values() if r["alignment_confidence"] is not None]
        summary = next((p for p in store.part_verdicts(cycle.end_ts - 60, cycle.end_ts + 600,
                                                       required_cameras=self.required)
                        if p["cycle_id"] == cycle_id and (p["cavity"] or 0) == cavity), None)
        prediction = store._query("SELECT risk, decision, likely_defect, model_kind FROM cycle_predictions "
                                  "WHERE cycle_id = ?", (cycle_id,))
        features = store.get_features(cycle_id) or {}
        return {
            "cycle_id": cycle_id, "cavity": cavity, "machine": cycle.machine_id, "sequence": cycle.sequence,
            "ts": cycle.end_ts, "verdict": summary["verdict"] if summary else "unknown",
            "defects": summary["defect_types"] if summary else [], "views": views,
            "link_confidence": round(float(np.mean(confidences)), 3) if confidences else None,
            "setpoints": cycle.setpoints, "scalars": cycle.scalars,
            "risk": dict(prediction[0]) if prediction else None,
            "deviations": self._deviations(features), "curves": self._curves(cycle),
            "audit": self._audit_for(cycle_id),
        }

    def _deviations(self, features: dict[str, float]) -> list[dict[str, Any]]:
        baseline = [c["features"] for c in self._cycles(BASELINE_CYCLES, newest=False)]
        out = []
        for name, (label, unit, setpoint) in SIGNALS.items():
            values = np.array([r[name] - (r.get(setpoint, 0.0) if setpoint else 0.0) for r in baseline if name in r])
            if name not in features or values.size < 20:
                continue
            center, sigma = _robust(values)
            offset = float(features.get(setpoint, 0.0)) if setpoint else 0.0
            value = float(features[name])
            z = (value - offset - center) / sigma
            direction, component = POINTS_TO.get(name, (0, None))
            out.append({"signal": name, "label": label, "unit": unit, "value": value, "expected": center + offset,
                        "z": round(z, 1), "abnormal": abs(z) >= WATCH_Z,
                        "points_to": component if component and z * direction >= WATCH_Z else None})
        return sorted(out, key=lambda d: -abs(d["z"]))

    def _curves(self, cycle) -> list[dict[str, Any]]:
        healthy = self._healthy_curves()
        out = []
        for name, (label, unit) in CURVES.items():
            values = cycle.curves.get(name)
            if values is None or len(values) < 2:
                continue
            x = np.linspace(0, 1, CURVE_POINTS)
            line = np.interp(x, np.linspace(0, 1, len(values)), values)
            out.append({"name": name, "label": label, "unit": unit, "seconds": len(values) / (cycle.sample_rate_hz or 1.0),
                        "values": [round(float(v), 2) for v in line], **healthy.get(name, {})})
        return out

    def _healthy_curves(self) -> dict[str, dict[str, list[float]]]:
        """Median and 1-99% band of the baseline cycles, cached: the baseline does not move."""
        if getattr(self, "_healthy", None) is not None:
            return self._healthy
        ids = [c["cycle_id"] for c in self._cycles(BASELINE_CYCLES, newest=False)]
        stacks: dict[str, list[np.ndarray]] = {}
        x = np.linspace(0, 1, CURVE_POINTS)
        for cycle_id in ids:
            cycle = self.store.get_cycle(cycle_id, with_curves=True)
            for name, values in (cycle.curves if cycle else {}).items():
                if len(values) >= 2:
                    stacks.setdefault(name, []).append(np.interp(x, np.linspace(0, 1, len(values)), values))
        out = {}
        for name, rows in stacks.items():
            m = np.vstack(rows)
            lo, mid, hi = np.percentile(m, [1, 50, 99], axis=0)
            out[name] = {"lo": np.round(lo, 2).tolist(), "mid": np.round(mid, 2).tolist(), "hi": np.round(hi, 2).tolist()}
        if len(ids) >= BASELINE_CYCLES:
            self._healthy = out
        return out

    # ------------------------------------------------------------------------------- process

    def spc(self, signal: str, limit: int = 240) -> dict[str, Any] | None:
        if signal not in SIGNALS:
            return None
        label, unit, setpoint = SIGNALS[signal]
        base = [c["features"] for c in self._cycles(BASELINE_CYCLES, newest=False)]
        base_values = np.array([r[signal] for r in base if signal in r])
        if base_values.size < 20:
            return {"signal": signal, "label": label, "unit": unit, "points": [], "learning": int(base_values.size)}
        center, sigma = _robust(base_values)
        ucl, lcl = center + 3 * sigma, center - 3 * sigma
        failed = self._failed_cycles()
        points, run_side, run = [], 0, 0
        for c in self._cycles(limit):
            v = c["features"].get(signal)
            if v is None:
                continue
            side = 1 if v > center else -1
            run = run + 1 if side == run_side else 1
            run_side = side
            rule = "beyond 3σ" if v > ucl or v < lcl else ("8 in a row on one side" if run >= 8 else None)
            points.append({"n": c["sequence"], "cycle_id": c["cycle_id"], "ts": c["ts"], "v": round(float(v), 3),
                           "rule": rule, "failed": c["cycle_id"] in failed})
        spec = INJECTION_MOLDING_SPEC_LIMITS.get(signal) if self.simulated else None
        cpk = None
        recent = np.array([p["v"] for p in points[-BASELINE_CYCLES:]])
        if spec and recent.size >= 30 and float(np.std(recent)) > 0:
            sides = [(spec["max"] - recent.mean()) if "max" in spec else None,
                     (recent.mean() - spec["min"]) if "min" in spec else None]
            cpk = round(float(min(s for s in sides if s is not None) / (3 * np.std(recent, ddof=1))), 2)
        return {"signal": signal, "label": label, "unit": unit, "center": center, "ucl": ucl, "lcl": lcl,
                "spec": {k: v for k, v in (spec or {}).items() if k in ("min", "max")}, "cpk": cpk, "points": points,
                "violations": sum(1 for p in points if p["rule"]), "baseline": int(base_values.size),
                "signals": [{"id": k, "label": v[0]} for k, v in SIGNALS.items()]}

    def _failed_cycles(self) -> set[str]:
        rows = self.store._query("SELECT DISTINCT cycle_id FROM inspections WHERE verdict = 'fail'") if self.store else []
        return {r["cycle_id"] for r in rows if r["cycle_id"]}

    def health(self) -> list[dict[str, Any]]:
        base = [c["features"] for c in self._cycles(BASELINE_CYCLES, newest=False)]
        rows = [c["features"] for c in self._cycles(600)]
        out = []
        for c in assess_machine(base, rows, self.components):
            h = c.history
            step = max(1, len(h) // 160)
            label = SIGNALS.get(c.signal, (c.signal, "", None))[0]
            out.append({"component": c.component, "signal": label, "wear": c.wear_index, "health": c.health,
                        "status": c.status, "cycles_to_act": c.cycles_to_act, "history": h[::step],
                        "watch": WATCH_Z, "act": ACT_Z})
        return out

    def risk(self, limit: int = 240) -> list[dict[str, Any]]:
        if self.store is None:
            return []
        rows = self.store._query(
            "SELECT c.cycle_id, c.sequence, c.end_ts, p.risk, p.decision, p.likely_defect, p.model_kind FROM process_cycles c "
            "JOIN cycle_predictions p ON p.cycle_id = c.cycle_id ORDER BY c.end_ts DESC LIMIT ?", (limit,))
        failed = self._failed_cycles()
        return [{"n": r["sequence"], "cycle_id": r["cycle_id"], "ts": float(r["end_ts"]), "risk": float(r["risk"]),
                 "decision": r["decision"], "likely": r["likely_defect"], "kind": r["model_kind"], "failed": r["cycle_id"] in failed}
                for r in reversed(rows)]

    # ------------------------------------------------------------------------------- quality

    def quality(self) -> dict[str, Any]:
        parts = sorted(self.parts(), key=lambda p: p["ts"])
        types: dict[str, int] = {}
        for p in parts:
            if p["verdict"] == "fail":
                for t in p["defect_types"] or ["unclassified"]:
                    types[t] = types.get(t, 0) + 1
        views = {c: {"camera": c, "judged": 0, "rejects": 0} for c in self.cameras}
        for p in parts:
            for camera, verdict in p["views"].items():
                if camera in views and verdict in ("pass", "fail"):
                    views[camera]["judged"] += 1
                    views[camera]["rejects"] += verdict == "fail"
        rolling, window = [], 30
        outcomes = [(p, p["verdict"] == "fail") for p in parts if p["verdict"] in ("pass", "fail")]
        for i in range(window, len(outcomes) + 1):
            chunk = outcomes[i - window:i]
            rolling.append({"n": i, "ts": chunk[-1][0]["ts"], "yield": 1 - sum(f for _, f in chunk) / window})
        scores = []
        if self.store is not None:
            scores = [float(r["anomaly_score"]) for r in self.store._query(
                "SELECT anomaly_score FROM inspections WHERE source='vision' AND anomaly_score IS NOT NULL "
                "ORDER BY ts DESC LIMIT 3000")]
        edges = np.linspace(0, 2.5, 51)
        counts = np.histogram(np.clip(scores, 0, 2.499), bins=edges)[0].tolist() if scores else []
        return {"pareto": sorted(({"type": k, "parts": v} for k, v in types.items()), key=lambda d: -d["parts"]),
                "views": list(views.values()), "rolling": rolling, "window": window,
                "scores": {"edges": edges.tolist(), "counts": counts, "n": len(scores)}}

    def model(self) -> dict[str, Any]:
        if self._model_cache is None:
            out: dict[str, Any] = {"path": self.model_path, "loaded": False}
            if self.model_path and Path(self.model_path).exists():
                from src.inference.model_registry import file_digest

                card = _read_json(Path(self.model_path).with_suffix(".card.json"))
                calibration = card.get("calibration") or {}
                if not calibration:
                    import torch

                    payload = torch.load(self.model_path, map_location="cpu", weights_only=True)
                    calibration = payload.get("calibration") or {}
                    card = {**card, **{k: payload.get(k) for k in ("kind", "threshold") if payload.get(k) is not None}}
                out = {"loaded": True, "name": Path(self.model_path).stem, "digest": file_digest(self.model_path),
                       "kind": card.get("kind"), "backbone": card.get("backbone"), "threshold": card.get("threshold"),
                       "part_frr_target": calibration.get("part_frr_target"), "n_views": calibration.get("n_views"),
                       "guaranteed": calibration.get("frr_guaranteed"), "n_holdout": calibration.get("n_holdout")}
            self._model_cache = out
        return self._model_cache

    # --------------------------------------------------------------------------------- audit

    def _audit_lines(self, limit: int) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in sorted(Path(self.audit_dir).glob("*.jsonl"), reverse=True):
            for line in reversed(path.read_text(encoding="utf-8", errors="replace").splitlines()):
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
                if len(out) >= limit:
                    return out
        return out

    def _audit_for(self, cycle_id: str) -> list[dict[str, Any]]:
        return [self._audit_row(r) for r in reversed(self._audit_lines(4000)) if r.get("cycle_id") == cycle_id]

    @staticmethod
    def _audit_row(r: dict[str, Any]) -> dict[str, Any]:
        return {"ts": r.get("ts"), "record": r.get("record", "view"), "cycle_id": r.get("cycle_id"),
                "camera": r.get("camera_id"), "verdict": r.get("verdict"), "score": r.get("normalized_score"),
                "image": (r.get("image_sha256") or "")[:12] or None,
                "model": ((r.get("models") or {}).get("anomaly") or {}).get("digest"),
                "hash": str(r.get("hash", ""))[:16], "prev": str(r.get("prev_hash", ""))[:16]}

    def audit(self, limit: int = 120) -> dict[str, Any]:
        return {"records": [self._audit_row(r) for r in self._audit_lines(limit)], "chain": self.verify()}

    def verify(self, max_age_s: float = 10.0) -> dict[str, Any]:
        stamp, cached = self._audit_cache
        if cached and time.monotonic() - stamp < max_age_s:
            return cached
        from src.inference.audit_log import verify_chain

        result: dict[str, Any] = {"ok": None, "records": 0}
        for _attempt in range(3):  # the runtime may be between its two writes
            if Path(self.audit_dir).exists():
                result = verify_chain(self.audit_dir, self.audit_key)
            if result.get("ok") is not False:
                break
            time.sleep(0.15)
        result = {**result, "checked": time.time()}
        self._audit_cache = (time.monotonic(), result)
        return result

    # ------------------------------------------------------------------------------- overview

    def prescriptions(self) -> list[dict[str, Any]]:
        if not Path(self.actions_db).exists():
            return []
        from src.operations.action_tracker import ActionTracker

        out = []
        for action in ActionTracker(sqlite_path=self.actions_db).list_actions(status="open", limit=20):
            payload = action.get("action_payload") or {}
            if payload.get("action_type") == "process_prescription":
                out.append({"id": action["action_id"], "text": payload.get("explanation"), "issued": action.get("timestamp")})
        return out

    def state(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        status = _read_json(self.status_path)
        age = now - float(status.get("timestamp") or 0.0)
        online = age < STATUS_STALE_S
        workers = status.get("workers") or {}
        plc = status.get("plc") or {}
        parts = sorted(self.parts(now=now), key=lambda p: p["ts"])
        counts = {v: sum(p["verdict"] == v for p in parts) for v in ("pass", "fail", "not_inspected", "incomplete")}
        judged = counts["pass"] + counts["fail"]
        recent = [p for p in parts if now - p["ts"] < 600]
        health = self.health()
        chain = self.verify()
        cameras = [{"id": c, "online": bool(online and c in workers and not workers[c].get("failed")
                                            and workers[c].get("capture_alive")),
                    "parts": (workers.get(c) or {}).get("parts_inspected"),
                    "restarts": (workers.get(c) or {}).get("restarts")} for c in self.cameras]
        links = [p for p in parts if p["cycle_id"]]
        latest = next((p for p in reversed(parts) if p["cycle_id"]), None)
        return {
            "now": now, "online": online, "simulated": self.simulated, "time_scale": self.time_scale,
            "machine": (self.cfg.get("process") or {}).get("machine_id", "imm_1"),
            "line": (self.cfg.get("process") or {}).get("line_id", "line"),
            "kpi": {"parts": len(parts), **counts, "yield": counts["pass"] / judged if judged else None,
                    "recent_yield": self._recent_yield(parts, 30), "rate_per_min": len(recent) / 10.0 if recent else 0.0,
                    "linked": len(links) / len(parts) if parts else None},
            "cameras": cameras, "health": health, "audit": chain, "model": self.model(),
            "plc": {**plc, "online": online}, "latest": self._key(latest) if latest else None,
            "alarms": self._alarms(online, age, plc, cameras, health, chain, parts),
        }

    @staticmethod
    def _recent_yield(parts: list[dict[str, Any]], n: int) -> float | None:
        judged = [p for p in parts if p["verdict"] in ("pass", "fail")][-n:]
        return sum(p["verdict"] == "pass" for p in judged) / len(judged) if len(judged) >= n else None

    def _alarms(self, online, age, plc, cameras, health, chain, parts) -> list[dict[str, str]]:
        out = []
        if not online:
            out.append({"level": "critical", "title": "Inspection runtime is offline",
                        "text": "No status received." if age > 1e6 else f"Last status {age:.0f} s ago."})
        elif plc.get("hold_reason"):
            out.append({"level": "critical", "title": "PLC heartbeat held", "text": str(plc["hold_reason"])})
        if online:
            down = [c["id"] for c in cameras if not c["online"]]
            if down:
                out.append({"level": "critical", "title": "Camera not delivering", "text": ", ".join(down)})
        if chain.get("ok") is False:
            out.append({"level": "critical", "title": "Audit trail broken", "text": str(chain.get("error"))})
        for c in health:
            if c["status"] != "OK":
                eta = f" About {c['cycles_to_act']} cycles to the act level at this rate." if c["cycles_to_act"] else ""
                out.append({"level": "critical" if c["status"] == "ACT" else "warning",
                            "title": f"{c['component']}: {'act now' if c['status'] == 'ACT' else 'drifting'}",
                            "text": f"{c['signal']} has moved {c['wear']:.1f}σ from its healthy baseline.{eta}"})
        judged = [p for p in parts if p["verdict"] in ("pass", "fail")][-30:]
        rejects = sum(p["verdict"] == "fail" for p in judged)
        if len(judged) == 30 and rejects >= 4:
            out.append({"level": "warning", "title": "Reject rate rising", "text": f"{rejects} of the last 30 parts failed."})
        missed = sum(p["verdict"] == "not_inspected" for p in parts[-30:])
        if missed:
            out.append({"level": "warning", "title": "Parts not inspected", "text": f"{missed} of the last 30 parts could not be judged."})
        return sorted(out, key=lambda a: a["level"] != "critical")
