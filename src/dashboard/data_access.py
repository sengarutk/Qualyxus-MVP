from __future__ import annotations

import json
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
