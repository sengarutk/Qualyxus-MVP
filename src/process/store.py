"""SQLite persistence for process cycles, inspections, risk predictions and models.

Lives in its own database file (``process.db`` next to the events DB by default) so
high-rate cycle ingestion never contends with the defect-event audit log.
"""
from __future__ import annotations

import io
import itertools
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.process.models import FAIL, PASS, InspectionRecord, ProcessCycle, RiskAssessment

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
    created_ts  REAL NOT NULL,
    complete    INTEGER NOT NULL DEFAULT 1
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
    active      INTEGER NOT NULL DEFAULT 0,
    machine_id  TEXT
);

CREATE TABLE IF NOT EXISTS cycle_gaps (
    machine_id  TEXT NOT NULL,
    ts          REAL NOT NULL,
    first_sequence INTEGER NOT NULL,
    last_sequence  INTEGER NOT NULL,
    missed      INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_gaps_machine_ts ON cycle_gaps(machine_id, ts);

CREATE TABLE IF NOT EXISTS baseline_resets (
    machine_id  TEXT NOT NULL,
    ts          REAL NOT NULL,
    kind        TEXT NOT NULL,
    reason      TEXT NOT NULL,
    by_whom     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_resets_machine_ts ON baseline_resets(machine_id, ts);
"""

# What a baseline reset covers: a repair makes the machine healthy again; a new mould or
# material also changes what a good cycle looks like, so the learned models start again.
MAINTENANCE, PROCESS_CHANGE = "maintenance", "process_change"
UNLINKED_GROUP_S = 5.0


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
        for column in ("pos_x REAL", "pos_y REAL", "pos_angle REAL", "cavity INTEGER NOT NULL DEFAULT 0"):
            self._add_column("inspections", column)
        self._add_column("process_cycles", "complete INTEGER NOT NULL DEFAULT 1")
        self._add_column("model_registry", "machine_id TEXT")
        self._conn.commit()

    def _add_column(self, table: str, definition: str) -> None:
        if definition.split()[0] in {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}:
            return
        try:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {definition}")
        except sqlite3.OperationalError as e:
            if "duplicate column" not in str(e):   # the console and the runtime may upgrade at once
                raise

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
                features, curves, sample_rate_hz, source, created_ts, complete
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cycle.cycle_id, cycle.machine_id, cycle.line_id, int(cycle.sequence),
                float(cycle.start_ts), float(cycle.end_ts),
                json.dumps(cycle.setpoints), json.dumps(cycle.scalars, default=str),
                json.dumps(features),
                _encode_curves(cycle.curves) if self.store_curves else None,
                float(cycle.sample_rate_hz), cycle.source, time.time(), int(cycle.complete),
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
            complete=bool(row["complete"]),
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

    def recent_cycles(self, limit: int = 200, *, with_curves: bool = False,
                      machine_id: str | None = None) -> list[ProcessCycle]:
        rows = self._query("SELECT * FROM process_cycles WHERE (? IS NULL OR machine_id = ?) "
                           "ORDER BY end_ts DESC LIMIT ?", (machine_id, machine_id, int(limit)))
        return [self._row_to_cycle(r, with_curves) for r in rows]

    def good_cycles(self, limit: int = 200, *, with_curves: bool = True, machine_id: str | None = None,
                    since_ts: float = 0.0, min_views: int = 1) -> list[ProcessCycle]:
        """Fully sampled cycles whose every inspection passed (most recent first)."""
        rows = self._query(
            """
            SELECT c.* FROM process_cycles c
            WHERE c.complete = 1 AND c.end_ts >= ? AND (? IS NULL OR c.machine_id = ?)
              AND (SELECT COUNT(DISTINCT i.camera_id) FROM inspections i
                   WHERE i.cycle_id = c.cycle_id AND i.verdict = ?) >= ?
              AND NOT EXISTS (SELECT 1 FROM inspections i WHERE i.cycle_id = c.cycle_id AND i.verdict != ?)
            ORDER BY c.end_ts DESC LIMIT ?
            """,
            (float(since_ts), machine_id, machine_id, PASS, max(1, int(min_views)), PASS, int(limit)),
        )
        return [self._row_to_cycle(r, with_curves) for r in rows]

    def last_sequence(self, machine_id: str) -> int | None:
        rows = self._query("SELECT sequence FROM process_cycles WHERE machine_id = ? ORDER BY end_ts DESC LIMIT 1",
                           (machine_id,))
        return int(rows[0]["sequence"]) if rows else None

    def machines(self) -> list[str]:
        return [r["machine_id"] for r in self._query("SELECT DISTINCT machine_id FROM process_cycles ORDER BY machine_id")]

    # ------------------------------------------------- lost cycles and baseline resets

    def save_gap(self, machine_id: str, ts: float, first_sequence: int, last_sequence: int) -> None:
        self._execute("INSERT INTO cycle_gaps VALUES (?, ?, ?, ?, ?)",
                      (machine_id, float(ts), int(first_sequence), int(last_sequence),
                       int(last_sequence) - int(first_sequence) + 1))

    def gaps_between(self, ts_from: float, ts_to: float) -> list[dict[str, Any]]:
        return [dict(r) for r in self._query("SELECT * FROM cycle_gaps WHERE ts BETWEEN ? AND ? ORDER BY ts",
                                             (ts_from, ts_to))]

    def reset_baseline(self, machine_id: str, *, kind: str = MAINTENANCE, reason: str, by: str,
                       ts: float | None = None) -> float:
        if kind not in (MAINTENANCE, PROCESS_CHANGE):
            raise ValueError(f"kind must be '{MAINTENANCE}' or '{PROCESS_CHANGE}', got {kind!r}")
        ts = time.time() if ts is None else float(ts)
        self._execute("INSERT INTO baseline_resets VALUES (?, ?, ?, ?, ?)", (machine_id, ts, kind, reason, by))
        return ts

    def baseline_start(self, machine_id: str | None, kind: str | None = None) -> float:
        """When this machine's current baseline began (0 = it was never reset)."""
        rows = self._query("SELECT MAX(ts) AS ts FROM baseline_resets WHERE (? IS NULL OR machine_id = ?) "
                           "AND (? IS NULL OR kind = ?)", (machine_id, machine_id, kind, kind))
        return float(rows[0]["ts"] or 0.0)

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

    def labelled_dataset(self, limit: int | None = None, *, min_confidence: float = 0.0,
                         machine_id: str | None = None, since_ts: float = 0.0,
                         required_cameras: Iterable[str] = ()) -> list[dict[str, Any]]:
        """One row per cycle with features and a quality label, for the newest ``limit`` cycles.

        FAIL if any part of the cycle failed. GOOD only if every part of it was really
        inspected: a view that could not judge, a required view that is missing, or a failure
        that may belong to another cycle (linked below ``min_confidence``) leaves the cycle
        without a label. Audits are ground truth: they decide, whatever vision said. Cycles
        that were not fully sampled are left out.
        """
        required = set(required_cameras)
        rows = self._query(
            """
            SELECT c.cycle_id, c.end_ts, c.features, i.camera_id, i.verdict, i.source, i.cavity, i.defect_type,
                   i.alignment_confidence AS conf
            FROM (SELECT cycle_id, end_ts, features FROM process_cycles p
                  WHERE complete = 1 AND end_ts >= ? AND (? IS NULL OR machine_id = ?)
                    AND EXISTS (SELECT 1 FROM inspections x WHERE x.cycle_id = p.cycle_id)
                  ORDER BY end_ts DESC LIMIT ?) c
            JOIN inspections i ON i.cycle_id = c.cycle_id
            ORDER BY c.end_ts, c.cycle_id
            """,
            (float(since_ts), machine_id, machine_id, int(limit) if limit else -1),
        )
        out = []
        for cycle_id, group in itertools.groupby(rows, key=lambda r: r["cycle_id"]):
            views = list(group)
            label = self._cycle_label(views, float(min_confidence), required)
            if label is None:
                continue
            types = [r["defect_type"] for r in views if r["verdict"] == FAIL and r["defect_type"]]
            out.append({"cycle_id": cycle_id, "end_ts": float(views[0]["end_ts"]),
                        "features": json.loads(views[0]["features"]), "label": label,
                        "defect_type": types[0] if (label and types) else None})
        return out

    @staticmethod
    def _cycle_label(views: list[sqlite3.Row], min_confidence: float, required: set[str]) -> int | None:
        audits = [r["verdict"] for r in views if r["source"] == "audit" and r["verdict"] in (PASS, FAIL)]
        if audits:
            return int(FAIL in audits)
        sure = [r for r in views if r["source"] != "audit" and (r["conf"] is None or r["conf"] >= min_confidence)]
        if any(r["verdict"] == FAIL for r in sure):
            return 1
        if any(r["verdict"] != PASS for r in views if r["source"] != "audit") or not sure:
            return None
        seen: dict[int, set[str]] = {}
        for r in sure:
            seen.setdefault(r["cavity"], set()).add(r["camera_id"])
        return None if any(required - cameras for cameras in seen.values()) else 0

    def part_verdicts(self, ts_from: float, ts_to: float, *,
                      required_cameras: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """One verdict per physical part (cycle + cavity), fused over every camera view of it.

        A part fails if any view (or an audit) failed it; an audit overrides vision. With
        ``required_cameras`` a part not yet seen by every required view is ``incomplete``:
        a passed part with a missing view has not actually been fully inspected. Views with no
        linked cycle that arrive together from different required cameras are one part. They
        can fail, so a failure never disappears, but nothing proves they saw the same part, so
        several cameras cannot pass it (the same rule the live verdict follows).
        """
        required = sorted(set(required_cameras or []))
        rows = self._query(
            "SELECT i.inspection_id, i.cycle_id, i.cavity, i.camera_id, i.verdict, i.defect_type, i.source, i.ts, "
            "i.alignment_confidence, c.machine_id FROM inspections i "
            "LEFT JOIN process_cycles c ON c.cycle_id = i.cycle_id WHERE i.ts BETWEEN ? AND ? ORDER BY i.ts",
            (ts_from, ts_to))
        parts: dict[Any, dict[str, Any]] = {}
        loose: dict[str, Any] | None = None     # the unlinked part still being grouped
        for r in rows:
            blank = {"cycle_id": r["cycle_id"], "cavity": r["cavity"], "ts": float(r["ts"]), "machine_id": r["machine_id"],
                     "views": {}, "audit": None, "defect_types": [], "confidences": []}
            if r["cycle_id"]:
                p = parts.setdefault((r["cycle_id"], r["cavity"]), blank)
            elif (len(required) > 1 and r["camera_id"] in required and loose is not None
                  and r["camera_id"] not in loose["views"] and float(r["ts"]) - loose["ts"] <= UNLINKED_GROUP_S):
                p = loose
            else:
                p = parts.setdefault((r["inspection_id"], None), blank)
                if r["camera_id"] in required:
                    loose = p
            if r["source"] == "audit":
                p["audit"] = r["verdict"]
            else:
                prev = p["views"].get(r["camera_id"])
                worst = [v for v in ("fail", "not_inspected", "pass") if v in (prev, r["verdict"])]
                p["views"][r["camera_id"]] = worst[0] if worst else r["verdict"]
                if r["alignment_confidence"] is not None:
                    p["confidences"].append(float(r["alignment_confidence"]))
            if r["verdict"] == "fail" and r["defect_type"]:
                p["defect_types"].append(r["defect_type"])
        out = []
        for p in parts.values():
            missing = [c for c in required if c not in p["views"]]
            seen = set(p["views"].values())
            vision = "fail" if "fail" in seen else ("not_inspected" if "not_inspected" in seen else "pass")
            if not p["views"]:
                vision = None               # an audit of a part inspected outside this window
            elif vision == "pass" and missing:
                vision = "incomplete"
            elif vision == "pass" and not p["cycle_id"] and len(required) > 1:
                vision = "not_inspected"
            out.append({"cycle_id": p["cycle_id"], "cavity": p["cavity"], "ts": p["ts"], "verdict": p["audit"] or vision,
                        "vision": vision, "views": p["views"], "machine_id": p["machine_id"],
                        "missing_views": missing, "audited": p["audit"] is not None, "confidences": p["confidences"],
                        "defect_types": sorted(set(p["defect_types"]))})
        return out

    def vision_verdicts(self, cycle_ids: Iterable[str], *,
                        required_cameras: Iterable[str] = ()) -> dict[tuple[str, int], str]:
        """What the cameras alone decided for every part of these cycles."""
        out: dict[tuple[str, int], str] = {}
        for cycle_id in cycle_ids:
            span = self._query("SELECT MIN(ts) AS a, MAX(ts) AS b FROM inspections WHERE cycle_id = ? "
                               "AND source != 'audit'", (cycle_id,))[0]
            if span["a"] is None:
                continue
            for p in self.part_verdicts(span["a"], span["b"], required_cameras=required_cameras):
                if p["cycle_id"] == cycle_id and p["vision"]:
                    out[(cycle_id, p["cavity"])] = p["vision"]
        return out

    def part_positions(self, limit: int = 600) -> list[sqlite3.Row]:
        """Measured part pose of the latest inspections, oldest first."""
        rows = self._query(
            "SELECT camera_id, ts, pos_x, pos_y, pos_angle FROM inspections "
            "WHERE pos_x IS NOT NULL ORDER BY ts DESC LIMIT ?", (int(limit),))
        return list(reversed(rows))

    def cycle_features(self, *, limit: int, newest: bool, machine_id: str | None = None,
                       since_ts: float = 0.0) -> list[dict[str, float]]:
        """Feature rows of the first or latest ``limit`` complete cycles, oldest first."""
        return [r["features"] for r in self.cycle_rows(limit=limit, newest=newest, machine_id=machine_id,
                                                       since_ts=since_ts)]

    def cycle_rows(self, *, limit: int, newest: bool, machine_id: str | None = None,
                   since_ts: float = 0.0) -> list[dict[str, Any]]:
        order = "DESC" if newest else "ASC"
        rows = self._query(
            "SELECT cycle_id, sequence, end_ts, features FROM process_cycles WHERE complete = 1 AND end_ts >= ? "
            f"AND (? IS NULL OR machine_id = ?) ORDER BY end_ts {order} LIMIT ?",
            (float(since_ts), machine_id, machine_id, int(limit)))
        out = [{"cycle_id": r["cycle_id"], "sequence": r["sequence"], "ts": float(r["end_ts"]),
                "features": json.loads(r["features"])} for r in rows]
        return list(reversed(out)) if newest else out

    def health_rows(self, machine_id: str, *, baseline: int = 100, recent: int = 600
                    ) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
        """(baseline, recent) cycles of one machine since its baseline was last reset."""
        start = self.baseline_start(machine_id)
        return (self.cycle_features(limit=baseline, newest=False, machine_id=machine_id, since_ts=start),
                self.cycle_features(limit=recent, newest=True, machine_id=machine_id, since_ts=start))

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
                       metrics: dict[str, Any], activate: bool, machine_id: str | None = None) -> None:
        with self._lock:
            if activate:
                self._conn.execute("UPDATE model_registry SET active = 0 WHERE machine_id IS ? OR machine_id IS NULL",
                                   (machine_id,))
            self._conn.execute(
                """
                INSERT OR REPLACE INTO model_registry
                    (version, created_ts, kind, path, n_samples, n_positive, metrics, active, machine_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (version, time.time(), kind, path, int(n_samples), int(n_positive),
                 json.dumps(metrics), 1 if activate else 0, machine_id),
            )
            self._conn.commit()

    def active_model(self, machine_id: str | None = None, *, any_machine: bool = True) -> dict[str, Any] | None:
        """The machine's active model. ``any_machine`` also accepts one saved before models were per machine."""
        rows = self._query(
            "SELECT * FROM model_registry WHERE active = 1 AND (? IS NULL OR machine_id = ? OR (? AND machine_id IS NULL)) "
            "ORDER BY machine_id IS NULL, created_ts DESC LIMIT 1", (machine_id, machine_id, int(any_machine)))
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
