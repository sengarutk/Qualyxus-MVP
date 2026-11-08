"""SQLite persistence for process cycles, inspections, risk predictions and models.

Lives in its own database file (``process.db`` next to the events DB by default) so
high-rate cycle ingestion never contends with the defect-event audit log.
"""
from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.process.models import FAIL, InspectionRecord, ProcessCycle, RiskAssessment

_SCHEMA = """
CREATE TABLE IF NOT EXISTS process_cycles (
    cycle_id    TEXT PRIMARY KEY,
    machine_id  TEXT NOT NULL,
    line_id     TEXT,
    sequence    INTEGER NOT NULL,
    start_ts    REAL NOT NULL,
    end_ts      REAL NOT NULL,
    setpoints   TEXT NOT NULL,
    scalars     TEXT NOT NULL,
    features    TEXT NOT NULL,
    curves      BLOB,
    sample_rate_hz REAL,
    source      TEXT,
    created_ts  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cycles_machine_end ON process_cycles(machine_id, end_ts);

CREATE TABLE IF NOT EXISTS inspections (
    inspection_id TEXT PRIMARY KEY,
    camera_id     TEXT NOT NULL,
    ts            REAL NOT NULL,
    verdict       TEXT NOT NULL,
    defect_type   TEXT,
    anomaly_score REAL,
    event_id      TEXT,
    cycle_id      TEXT,
    alignment_confidence REAL,
    source        TEXT NOT NULL DEFAULT 'vision',
    pos_x         REAL,
    pos_y         REAL,
    pos_angle     REAL,
    cavity        INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_inspections_cycle ON inspections(cycle_id);
CREATE INDEX IF NOT EXISTS idx_inspections_ts ON inspections(ts);

CREATE TABLE IF NOT EXISTS cycle_predictions (
    cycle_id      TEXT PRIMARY KEY,
    ts            REAL NOT NULL,
    risk          REAL NOT NULL,
    decision      TEXT NOT NULL,
    model_version TEXT NOT NULL,
    model_kind    TEXT NOT NULL,
    likely_defect TEXT,
    likely_defect_probability REAL,
    top_factors   TEXT
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON cycle_predictions(ts);

CREATE TABLE IF NOT EXISTS model_registry (
    version     TEXT PRIMARY KEY,
    created_ts  REAL NOT NULL,
    kind        TEXT NOT NULL,
    path        TEXT NOT NULL,
    n_samples   INTEGER NOT NULL,
    n_positive  INTEGER NOT NULL,
    metrics     TEXT NOT NULL,
    active      INTEGER NOT NULL DEFAULT 0
);
"""


def _encode_curves(curves: dict[str, np.ndarray]) -> bytes | None:
    if not curves:
        return None
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{k: np.asarray(v, dtype=np.float32) for k, v in curves.items()})
    return buffer.getvalue()


def _decode_curves(blob: bytes | None) -> dict[str, np.ndarray]:
    if not blob:
        return {}
    with np.load(io.BytesIO(blob)) as data:
        return {k: data[k] for k in data.files}


class ProcessStore:
    def __init__(self, db_path: str | Path = "data/events/process.db", *, store_curves: bool = True):
        self.db_path = str(db_path)
        self.store_curves = bool(store_curves)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # One long-lived connection: opening/closing per statement forces a WAL checkpoint
        # (and fsync) on every close, which costs ~20 ms per write.
        self._conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(_SCHEMA)
        columns = {r[1] for r in self._conn.execute("PRAGMA table_info(inspections)")}
        for column in ("pos_x", "pos_y", "pos_angle"):
            if column not in columns:
                self._conn.execute(f"ALTER TABLE inspections ADD COLUMN {column} REAL")
        if "cavity" not in columns:
            self._conn.execute("ALTER TABLE inspections ADD COLUMN cavity INTEGER NOT NULL DEFAULT 0")
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    # ---------------------------------------------------------------- cycles

    def save_cycle(self, cycle: ProcessCycle, features: dict[str, float]) -> None:
        self._execute(
            """
            INSERT OR REPLACE INTO process_cycles (
                cycle_id, machine_id, line_id, sequence, start_ts, end_ts, setpoints, scalars,
                features, curves, sample_rate_hz, source, created_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cycle.cycle_id, cycle.machine_id, cycle.line_id, int(cycle.sequence),
                float(cycle.start_ts), float(cycle.end_ts),
                json.dumps(cycle.setpoints), json.dumps(cycle.scalars, default=str),
                json.dumps(features),
                _encode_curves(cycle.curves) if self.store_curves else None,
                float(cycle.sample_rate_hz), cycle.source, time.time(),
            ),
        )

    @staticmethod
    def _row_to_cycle(row: sqlite3.Row, with_curves: bool) -> ProcessCycle:
        return ProcessCycle(
            cycle_id=row["cycle_id"],
            machine_id=row["machine_id"],
            line_id=row["line_id"],
            sequence=int(row["sequence"]),
            start_ts=float(row["start_ts"]),
            end_ts=float(row["end_ts"]),
            setpoints=json.loads(row["setpoints"]),
            scalars=json.loads(row["scalars"]),
            curves=_decode_curves(row["curves"]) if with_curves else {},
            sample_rate_hz=float(row["sample_rate_hz"] or 50.0),
            source=row["source"] or "unknown",
        )

    def get_cycle(self, cycle_id: str, *, with_curves: bool = False) -> ProcessCycle | None:
        rows = self._query("SELECT * FROM process_cycles WHERE cycle_id = ?", (cycle_id,))
        return self._row_to_cycle(rows[0], with_curves) if rows else None

    def get_features(self, cycle_id: str) -> dict[str, float] | None:
        rows = self._query("SELECT features FROM process_cycles WHERE cycle_id = ?", (cycle_id,))
        return json.loads(rows[0]["features"]) if rows else None

    def cycles_between(self, machine_id: str, end_from: float, end_to: float) -> list[tuple[str, float]]:
        rows = self._query(
            "SELECT cycle_id, end_ts FROM process_cycles WHERE machine_id = ? AND end_ts BETWEEN ? AND ? ORDER BY end_ts",
            (machine_id, float(end_from), float(end_to)),
        )
        return [(r["cycle_id"], float(r["end_ts"])) for r in rows]

    def cycles_ending_before(self, machine_id: str, ts: float, limit: int) -> list[tuple[str, float]]:
        rows = self._query(
            "SELECT cycle_id, end_ts FROM process_cycles WHERE machine_id = ? AND end_ts <= ? ORDER BY end_ts DESC LIMIT ?",
            (machine_id, float(ts), int(limit)),
        )
        return [(r["cycle_id"], float(r["end_ts"])) for r in rows]

    def recent_cycles(self, limit: int = 200, *, with_curves: bool = False) -> list[ProcessCycle]:
        rows = self._query("SELECT * FROM process_cycles ORDER BY end_ts DESC LIMIT ?", (int(limit),))
        return [self._row_to_cycle(r, with_curves) for r in rows]

    def good_cycles(self, limit: int = 200, *, with_curves: bool = True) -> list[ProcessCycle]:
        """Cycles whose linked inspections all passed (most recent first)."""
        rows = self._query(
            """
            SELECT c.* FROM process_cycles c
            WHERE EXISTS (SELECT 1 FROM inspections i WHERE i.cycle_id = c.cycle_id)
              AND NOT EXISTS (SELECT 1 FROM inspections i WHERE i.cycle_id = c.cycle_id AND i.verdict = ?)
            ORDER BY c.end_ts DESC LIMIT ?
            """,
            (FAIL, int(limit)),
        )
        return [self._row_to_cycle(r, with_curves) for r in rows]

    def count_cycles(self) -> int:
        return int(self._query("SELECT COUNT(*) AS n FROM process_cycles")[0]["n"])

    # ----------------------------------------------------------- inspections

    def save_inspection(self, record: InspectionRecord) -> InspectionRecord:
        if not record.inspection_id:
            record.inspection_id = uuid.uuid4().hex
        self._execute(
            """
            INSERT OR REPLACE INTO inspections (
                inspection_id, camera_id, ts, verdict, defect_type, anomaly_score, event_id,
                cycle_id, alignment_confidence, source, pos_x, pos_y, pos_angle, cavity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.inspection_id, record.camera_id, float(record.ts), record.verdict,
                record.defect_type, record.anomaly_score, record.event_id, record.cycle_id,
                record.alignment_confidence, record.source, record.pos_x, record.pos_y, record.pos_angle,
                record.cavity,
            ),
        )
        return record

    def link_counts(self, camera_id: str, cycle_ids: Iterable[str]) -> dict[str, int]:
        """Parts this camera has already linked to each cycle."""
        ids = list(cycle_ids)
        if not ids:
            return {}
        marks = ",".join("?" for _ in ids)
        rows = self._query(
            f"SELECT cycle_id, COUNT(*) AS n FROM inspections WHERE camera_id = ? AND cycle_id IN ({marks}) "
            "GROUP BY cycle_id",
            [camera_id, *ids],
        )
        return {r["cycle_id"]: int(r["n"]) for r in rows}

    def labelled_dataset(self, limit: int | None = None, *, min_confidence: float = 0.0) -> list[dict[str, Any]]:
        """One row per cycle with features and a quality label from its inspections.

        A cycle is labelled FAIL if any linked vision inspection or audit failed. Audits
        override vision verdicts for the same cycle (they are the ground truth). A vision
        result linked below ``min_confidence`` may belong to another cycle, so it is left
        out rather than teaching the risk model a wrong label.
        """
        sql = """
            SELECT c.cycle_id, c.end_ts, c.features,
                   MAX(CASE WHEN i.source = 'audit' THEN 1 ELSE 0 END) AS has_audit,
                   MAX(CASE WHEN i.source = 'audit' AND i.verdict = 'fail' THEN 1 ELSE 0 END) AS audit_fail,
                   MAX(CASE WHEN i.source = 'vision' AND i.verdict = 'fail' THEN 1 ELSE 0 END) AS vision_fail,
                   GROUP_CONCAT(CASE WHEN i.verdict = 'fail' THEN i.defect_type END) AS defect_types
            FROM process_cycles c JOIN inspections i ON i.cycle_id = c.cycle_id
                 AND i.verdict IN ('pass', 'fail')
                 AND (i.source = 'audit' OR i.alignment_confidence IS NULL OR i.alignment_confidence >= ?)
            GROUP BY c.cycle_id
            ORDER BY c.end_ts
        """
        if limit:
            sql += f" LIMIT {int(limit)}"
        out = []
        for r in self._query(sql, (float(min_confidence),)):
            label = int(r["audit_fail"]) if int(r["has_audit"]) else int(r["vision_fail"])
            types = [t for t in (r["defect_types"] or "").split(",") if t]
            out.append({
                "cycle_id": r["cycle_id"],
                "end_ts": float(r["end_ts"]),
                "features": json.loads(r["features"]),
                "label": label,
                "defect_type": types[0] if (label and types) else None,
            })
        return out

    def part_verdicts(self, ts_from: float, ts_to: float, *,
                      required_cameras: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """One verdict per physical part (cycle + cavity), fused over every camera view of it.

        A part fails if any view (or an audit) failed it; an audit overrides vision. With
        ``required_cameras`` a part not yet seen by every required view is ``incomplete``:
        a passed part with a missing view has not actually been fully inspected. An
        inspection with no linked cycle is still a part, so a failure can never disappear.
        """
        required = sorted(set(required_cameras or []))
        rows = self._query(
            "SELECT inspection_id, cycle_id, cavity, camera_id, verdict, defect_type, source, ts FROM inspections "
            "WHERE ts BETWEEN ? AND ? ORDER BY ts", (ts_from, ts_to))
        parts: dict[Any, dict[str, Any]] = {}
        for r in rows:
            key = (r["cycle_id"], r["cavity"]) if r["cycle_id"] else (r["inspection_id"], None)
            p = parts.setdefault(key, {"cycle_id": r["cycle_id"], "cavity": r["cavity"], "ts": float(r["ts"]),
                                       "views": {}, "audit": None, "defect_types": []})
            if r["source"] == "audit":
                p["audit"] = r["verdict"]
            else:
                prev = p["views"].get(r["camera_id"])
                worst = [v for v in ("fail", "not_inspected", "pass") if v in (prev, r["verdict"])]
                p["views"][r["camera_id"]] = worst[0] if worst else r["verdict"]
            if r["verdict"] == "fail" and r["defect_type"]:
                p["defect_types"].append(r["defect_type"])
        out = []
        for p in parts.values():
            missing = [c for c in required if c not in p["views"]]
            seen = set(p["views"].values())
            vision = "fail" if "fail" in seen else ("not_inspected" if "not_inspected" in seen else "pass")
            verdict = p["audit"] or vision
            if verdict == "pass" and missing and p["audit"] is None:
                verdict = "incomplete"
            out.append({"cycle_id": p["cycle_id"], "cavity": p["cavity"], "ts": p["ts"], "verdict": verdict,
                        "views": p["views"],
                        "missing_views": missing, "audited": p["audit"] is not None,
                        "defect_types": sorted(set(p["defect_types"]))})
        return out

    def part_positions(self, limit: int = 600) -> list[sqlite3.Row]:
        """Measured part pose of the latest inspections, oldest first."""
        rows = self._query(
            "SELECT camera_id, ts, pos_x, pos_y, pos_angle FROM inspections "
            "WHERE pos_x IS NOT NULL ORDER BY ts DESC LIMIT ?", (int(limit),))
        return list(reversed(rows))

    def cycle_features(self, *, limit: int, newest: bool) -> list[dict[str, float]]:
        """Feature rows of the first or latest ``limit`` cycles, oldest first."""
        order = "DESC" if newest else "ASC"
        rows = self._query(f"SELECT features FROM process_cycles ORDER BY end_ts {order} LIMIT ?", (int(limit),))
        out = [json.loads(r["features"]) for r in rows]
        return list(reversed(out)) if newest else out

    def inspections_between(self, ts_from: float, ts_to: float) -> list[sqlite3.Row]:
        return self._query("SELECT * FROM inspections WHERE ts BETWEEN ? AND ? ORDER BY ts", (ts_from, ts_to))

    # ----------------------------------------------------------- predictions

    def save_prediction(self, assessment: RiskAssessment) -> None:
        self._execute(
            """
            INSERT OR REPLACE INTO cycle_predictions (
                cycle_id, ts, risk, decision, model_version, model_kind, likely_defect,
                likely_defect_probability, top_factors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assessment.cycle_id, float(assessment.ts or time.time()), float(assessment.risk),
                assessment.decision, assessment.model_version, assessment.model_kind,
                assessment.likely_defect, assessment.likely_defect_probability,
                json.dumps(assessment.top_factors),
            ),
        )

    def predictions_between(self, ts_from: float, ts_to: float) -> list[sqlite3.Row]:
        return self._query("SELECT * FROM cycle_predictions WHERE ts BETWEEN ? AND ? ORDER BY ts", (ts_from, ts_to))

    # ---------------------------------------------------------------- models

    def register_model(self, version: str, kind: str, path: str, n_samples: int, n_positive: int,
                       metrics: dict[str, Any], activate: bool) -> None:
        with self._lock:
            if activate:
                self._conn.execute("UPDATE model_registry SET active = 0")
            self._conn.execute(
                """
                INSERT OR REPLACE INTO model_registry
                    (version, created_ts, kind, path, n_samples, n_positive, metrics, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (version, time.time(), kind, path, int(n_samples), int(n_positive),
                 json.dumps(metrics), 1 if activate else 0),
            )
            self._conn.commit()

    def active_model(self) -> dict[str, Any] | None:
        rows = self._query("SELECT * FROM model_registry WHERE active = 1 ORDER BY created_ts DESC LIMIT 1")
        if not rows:
            return None
        row = dict(rows[0])
        row["metrics"] = json.loads(row["metrics"])
        return row

    def models(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self._query("SELECT * FROM model_registry ORDER BY created_ts DESC LIMIT ?", (int(limit),))
        return [{**dict(r), "metrics": json.loads(r["metrics"])} for r in rows]

    # ------------------------------------------------------------- retention

    def prune_curves(self, older_than_s: float) -> int:
        """Drop raw curves (keep features) for cycles older than the given age."""
        cutoff = time.time() - float(older_than_s)
        cur = self._execute("UPDATE process_cycles SET curves = NULL WHERE end_ts < ? AND curves IS NOT NULL", (cutoff,))
        return int(cur.rowcount)
