from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from src.analytics.esg_engine import ESGEngine
from src.config.config import Config


def _shift_from_hour(hour: int) -> str:
    if 6 <= hour < 14:
        return "A"
    if 14 <= hour < 22:
        return "B"
    return "C"


@st.cache_resource
def get_engine(sqlite_path: str):
    # NOTE: dashboard uses its own cached engine (separate from runtime logger's cache).
    # This is intentional: Streamlit is a separate process and should manage its own pool.
    return create_engine(f"sqlite:///{sqlite_path}", future=True)


def events_table_ready(sqlite_path: str) -> bool:
    """False until the edge engine has created the events database."""
    if not Path(sqlite_path).exists():
        return False
    with sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True) as conn:
        row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'defect_events'").fetchone()
    return row is not None


def _read_status(status_path: str) -> dict:
    p = Path(status_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(ttl=10)
def get_kpi_metrics(sqlite_path: str, status_path: str) -> dict:
    engine = get_engine(sqlite_path)
    now = time.localtime()
    current_shift = _shift_from_hour(now.tm_hour)

    with engine.connect() as conn:
        # Event timestamps are stored in UTC; the A/B/C shift calendar is plant-local time.
        # (shift_id carries telemetry labels such as "day"/"night", so it cannot be compared
        # with the A/B/C calendar used here.)
        defects_per_hour = conn.execute(
            text(
                """
                WITH enriched AS (
                    SELECT
                        CASE
                            WHEN CAST(strftime('%H', timestamp, 'localtime') AS INTEGER) BETWEEN 6 AND 13 THEN 'A'
                            WHEN CAST(strftime('%H', timestamp, 'localtime') AS INTEGER) BETWEEN 14 AND 21 THEN 'B'
                            ELSE 'C'
                        END AS shift_name,
                        timestamp
                    FROM defect_events
                )
                SELECT COUNT(*) AS c
                FROM enriched
                WHERE shift_name = :shift_name
                  AND datetime(timestamp) >= datetime('now', '-1 hour')
                """
            ),
            {"shift_name": current_shift},
        ).scalar() or 0

        avg_conf = conn.execute(text("SELECT AVG(confidence) FROM defect_events")).scalar()
        top_defect = conn.execute(
            text(
                """
                SELECT defect_type, COUNT(*) AS c
                FROM defect_events
                GROUP BY defect_type
                ORDER BY c DESC
                LIMIT 1
                """
            )
        ).first()

    status = _read_status(status_path)
    workers = status.get("workers", {}) if isinstance(status, dict) else {}
    active_cameras = sum(1 for _, w in workers.items() if w.get("capture_alive"))

    started_ats = [float(w.get("started_at", 0.0)) for _, w in workers.items() if w.get("started_at")]
    uptime_seconds = max(0.0, time.time() - min(started_ats)) if started_ats else 0.0

    return {
        "defects_per_hour": int(defects_per_hour),
        "avg_confidence": float(avg_conf or 0.0),
        "top_defect_type": (top_defect[0] if top_defect else "n/a"),
        "active_cameras": int(active_cameras),
        "system_uptime_seconds": uptime_seconds,
        "current_shift": current_shift,
    }


@st.cache_data(ttl=10)
def query_defects_per_hour(sqlite_path: str, lookback_hours: int = 24) -> pd.DataFrame:
    engine = get_engine(sqlite_path)
    sql = text(
        """
        SELECT
            strftime('%Y-%m-%d %H:00:00', timestamp) AS hour_bucket,
            COUNT(*) AS defect_count
        FROM defect_events
        WHERE datetime(timestamp) >= datetime('now', :lookback)
        GROUP BY hour_bucket
        ORDER BY hour_bucket
        """
    )
    return pd.read_sql_query(sql, engine, params={"lookback": f"-{lookback_hours} hour"})


@st.cache_data(ttl=10)
def query_severity_distribution(sqlite_path: str) -> pd.DataFrame:
    engine = get_engine(sqlite_path)
    sql = text(
        """
        SELECT severity, COUNT(*) AS count
        FROM defect_events
        GROUP BY severity
        ORDER BY severity
        """
    )
    return pd.read_sql_query(sql, engine)


@st.cache_data(ttl=10)
def query_defects_by_shift(sqlite_path: str) -> pd.DataFrame:
    engine = get_engine(sqlite_path)
    sql = text(
        """
        WITH enriched AS (
            SELECT
                COALESCE(
                    shift_id,
                    CASE
                        WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 6 AND 13 THEN 'A'
                        WHEN CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN 14 AND 21 THEN 'B'
                        ELSE 'C'
                    END
                ) AS shift_name,
                severity
            FROM defect_events
        )
        SELECT shift_name AS shift, severity, COUNT(*) AS count
        FROM enriched
        GROUP BY shift_name, severity
        ORDER BY shift_name, severity
        """
    )
    return pd.read_sql_query(sql, engine)


@st.cache_data(ttl=10)
def query_defects_by_machine(sqlite_path: str) -> pd.DataFrame:
    engine = get_engine(sqlite_path)
    sql = text(
        """
        SELECT COALESCE(machine_id, source, 'unknown') AS machine, COUNT(*) AS count
        FROM defect_events
        GROUP BY COALESCE(machine_id, source, 'unknown')
        ORDER BY count DESC
        """
    )
    return pd.read_sql_query(sql, engine)


@st.cache_data(ttl=10)
def query_hourly_heatmap(sqlite_path: str, lookback_days: int = 7) -> pd.DataFrame:
    engine = get_engine(sqlite_path)
    sql = text(
        """
        SELECT
            strftime('%w', timestamp) AS day_of_week,
            strftime('%H', timestamp) AS hour_of_day,
            COUNT(*) AS count
        FROM defect_events
        WHERE datetime(timestamp) >= datetime('now', :lookback)
        GROUP BY day_of_week, hour_of_day
        ORDER BY day_of_week, hour_of_day
        """
    )
    return pd.read_sql_query(sql, engine, params={"lookback": f"-{lookback_days} day"})


@st.cache_data(ttl=10)
def query_timeline_events(sqlite_path: str, limit: int = 300) -> pd.DataFrame:
    engine = get_engine(sqlite_path)
    sql = text(
        """
        SELECT
            event_id, timestamp, source, defect_type, confidence, severity,
            machine_id, shift_id, full_image_path, overlay_path, crop_path, vlm_narrative
        FROM defect_events
        ORDER BY datetime(timestamp) DESC
        LIMIT :limit
        """
    )
    return pd.read_sql_query(sql, engine, params={"limit": int(limit)})


@st.cache_data(ttl=10)
def query_live_events(sqlite_path: str, limit: int = 50) -> pd.DataFrame:
    return query_timeline_events(sqlite_path=sqlite_path, limit=limit)


def _calculate_delta(current: float, previous: float) -> float:
    return round(float(current) - float(previous), 2)


@st.cache_data(ttl=300)
def fetch_esg_data(sqlite_path: str) -> dict:
    engine = get_engine(sqlite_path)
    cfg = Config.load()
    esg_engine = ESGEngine(config=cfg)

    current_sql = text(
        """
        SELECT *
        FROM defect_events
        WHERE datetime(timestamp) >= datetime('now', '-24 hour')
        """
    )
    previous_sql = text(
        """
        SELECT *
        FROM defect_events
        WHERE datetime(timestamp) >= datetime('now', '-48 hour')
          AND datetime(timestamp) < datetime('now', '-24 hour')
        """
    )

    current_df = pd.read_sql_query(current_sql, engine)
    previous_df = pd.read_sql_query(previous_sql, engine)

    current = esg_engine.get_impact_metrics(current_df)
    previous = esg_engine.get_impact_metrics(previous_df)

    return {
        "current": current,
        "previous": previous,
        "delta": {
            "scrap_prevented_kg": _calculate_delta(
                current.get("scrap_prevented_kg", 0.0), previous.get("scrap_prevented_kg", 0.0)
            ),
            "revenue_recovered_usd": _calculate_delta(
                current.get("revenue_recovered_usd", 0.0), previous.get("revenue_recovered_usd", 0.0)
            ),
            "co2_emissions_offset_kg": _calculate_delta(
                current.get("co2_emissions_offset_kg", 0.0), previous.get("co2_emissions_offset_kg", 0.0)
            ),
        },
    }


@st.cache_data(ttl=60)
def fetch_federated_status() -> dict:
    cfg = Config.load()
    status_path = Path(cfg.get("paths", "data_dir", default="data")) / "runtime" / "federated_status.json"
    if not status_path.exists():
        return {
            "collaborative_contribution_count": 0,
            "global_knowledge_version": "n/a",
            "privacy_tooltip": "Qualyxus Federated Learning ensures no raw images ever leave your facility. Only encrypted intelligence is shared with the Global Network.",
        }
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload.setdefault("collaborative_contribution_count", 0)
    payload.setdefault("global_knowledge_version", "n/a")
    payload.setdefault(
        "privacy_tooltip",
        "Qualyxus Federated Learning ensures no raw images ever leave your facility. Only encrypted intelligence is shared with the Global Network.",
    )
    return payload


@st.cache_data(ttl=5)
def fetch_system_health() -> dict:
    cfg = Config.load()
    path = Path(cfg.get("paths", "data_dir", default="data")) / "runtime" / "system_health.json"
    if not path.exists():
        return {"state": "SAFE", "vitals": {}, "resource_log": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.setdefault("state", "SAFE")
        payload.setdefault("vitals", {})
        payload.setdefault("resource_log", [])
        return payload
    except Exception:
        return {"state": "SAFE", "vitals": {}, "resource_log": []}


@st.cache_data(ttl=10)
def fetch_process_overview(process_db_path: str, actions_db_path: str, window_hours: float = 8.0) -> dict:
    """Process-quality loop: KPIs, risk timeline and pending prescriptions."""
    if not Path(process_db_path).exists():
        return {}
    from src.process.kpis import compute_kpis
    from src.process.store import ProcessStore

    store = ProcessStore(process_db_path)
    now = time.time()
    since = now - window_hours * 3600.0
    kpis = compute_kpis(store, since, now)
    model = store.active_model()

    engine = create_engine(f"sqlite:///{process_db_path}", future=True)
    timeline = pd.read_sql_query(
        text(
            """
            SELECT p.cycle_id, p.ts, p.risk, p.decision, p.likely_defect, p.top_factors,
                   (SELECT MAX(CASE WHEN i.verdict = 'fail' THEN 1 ELSE 0 END)
                      FROM inspections i WHERE i.cycle_id = p.cycle_id) AS failed
            FROM cycle_predictions p
            WHERE p.ts >= :since
            ORDER BY p.ts
            """
        ),
        engine,
        params={"since": since},
    )
    engine.dispose()
    if not timeline.empty:
        timeline["time"] = pd.to_datetime(timeline["ts"], unit="s")
        timeline["factors"] = timeline["top_factors"].map(
            lambda raw: ", ".join(f"{f['feature']} {f['direction']} normal" for f in json.loads(raw or "[]")[:3])
        )

    prescriptions: list[dict] = []
    if Path(actions_db_path).exists():
        from src.operations.action_tracker import ActionTracker

        tracker = ActionTracker(sqlite_path=actions_db_path)
        for status in ("open", "in_progress"):
            for action in tracker.list_actions(status=status, limit=100):
                payload = action.get("action_payload") or {}
                if payload.get("action_type") == "process_prescription":
                    prescriptions.append({
                        "action_id": action["action_id"],
                        "status": action["status"],
                        "approved_by": action.get("approved_by"),
                        "recommendation": payload.get("explanation"),
                        "issued": action.get("timestamp"),
                    })
    return {"kpis": kpis, "model": model, "timeline": timeline, "prescriptions": prescriptions}


@st.cache_data(ttl=15)
def fetch_vision_overview(status_path: str, process_db_path: str, registry_dir: str | None, review_dir: str,
                          audit_dir: str, window_hours: float = 8.0, required_cameras: tuple[str, ...] = ()) -> dict:
    """Vision-model operations: health per camera, registry/shadow, review queue, defect mix, audit."""
    out: dict = {"cameras": [], "registry": [], "requests": [], "shadow": [], "review": {}, "defect_types": [],
                 "parts": {}, "audit": None}
    status = _read_status(status_path)
    cams = status.get("workers", {}) if isinstance(status, dict) else {}
    for cam_id, cam in (cams or {}).items():
        if not isinstance(cam, dict) or "parts_inspected" not in cam:
            continue
        health = cam.get("model_health") or {}
        out["cameras"].append({
            "camera": cam_id, "recipe": health.get("recipe") or cam.get("recipe"),
            "models": ", ".join(f"{k}={v}" for k, v in (health.get("models") or {}).items()),
            "parts": cam.get("parts_inspected", 0), "failed": cam.get("parts_failed", 0),
            "not_inspected": cam.get("parts_not_inspected", 0),
            "reject_rate": health.get("reject_rate"), "score_drift": health.get("score_drift"),
            "alerts": " | ".join(health.get("alerts") or []),
        })
    if registry_dir and Path(registry_dir).exists():
        from src.inference.model_registry import VisionModelRegistry

        reg = VisionModelRegistry(registry_dir)
        for row in reg.overview():
            last = row.get("last_change") or {}
            out["registry"].append({**{k: row[k] for k in ("recipe", "role", "active", "shadow", "versions")},
                                    "last_change": f"{last.get('action')} {last.get('version')} by {last.get('by')}"})
            if row["shadow"] and row["role"] == "anomaly":
                out["shadow"].append(reg.shadow_report(row["recipe"], "anomaly", review_dir=review_dir))
        out["requests"] = [{k: r.get(k) for k in ("id", "recipe", "role", "camera_id", "reason", "status", "decided_by")}
                           for r in reg.requests() if r["status"] in ("pending", "approved")]
    if Path(review_dir).exists():
        from src.inference.labeling import prioritise, review_items

        everything = review_items(review_dir, include_reviewed=True)
        pending = [i for i in everything if not (i.get("review") or {}).get("verdict")]
        reviewed = [i for i in everything if (i.get("review") or {}).get("verdict")]
        wrong = sum(1 for i in reviewed if i["review"]["verdict"] != i.get("model_verdict"))
        out["review"] = {
            "pending": len(pending), "reviewed": len(reviewed),
            "model_agreement": round(1 - wrong / len(reviewed), 4) if reviewed else None,
            "next": [{"id": i["id"], "camera": i.get("camera_id"), "score": i.get("normalized_score"),
                      "model_verdict": i.get("model_verdict"), "predicted_type": i.get("predicted_type")}
                     for i in prioritise(pending)[:15]],
        }
    if Path(process_db_path).exists():
        from src.process.store import ProcessStore

        store = ProcessStore(process_db_path)
        try:
            now = time.time()
            parts = store.part_verdicts(now - window_hours * 3600.0, now, required_cameras=required_cameras)
            counts = pd.Series([p["verdict"] for p in parts]).value_counts().to_dict() if parts else {}
            out["parts"] = {"total": len(parts), **{str(k): int(v) for k, v in counts.items()},
                            "multi_view": sum(1 for p in parts if len(p["views"]) > 1)}
            out["recent_parts"] = [
                {"part": p["cycle_id"], "verdict": p["verdict"], "views_seen": len(p["views"]),
                 "failed_views": ", ".join(c for c, v in p["views"].items() if v == "fail"),
                 "missing_views": ", ".join(p["missing_views"]),
                 "defect": ", ".join(sorted(set(p["defect_types"])))}
                for p in sorted(parts, key=lambda p: p["ts"], reverse=True)[:15]
            ]
            types = pd.Series([t for p in parts if p["verdict"] == "fail" for t in (p["defect_types"] or ["unspecified"])])
            if not types.empty:
                out["defect_types"] = [{"defect_type": k, "parts": int(v)} for k, v in types.value_counts().items()]
        finally:
            store.close()
    if Path(audit_dir).exists():
        from src.inference.audit_log import verify_chain

        out["audit"] = verify_chain(audit_dir)
    return out
