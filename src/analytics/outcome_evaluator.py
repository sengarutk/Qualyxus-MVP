from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


DEFAULT_EVENTS_DB = "data/events/events.db"
DEFAULT_ACTIONS_DB = "data/events/actions.db"
VALID_6M = {"Man", "Machine", "Method", "Material", "Measurement", "Environment"}


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


@dataclass
class WindowStats:
    start: datetime
    end: datetime
    event_count: int

    @property
    def duration_hours(self) -> float:
        return max((self.end - self.start).total_seconds() / 3600.0, 1e-9)

    @property
    def defect_rate_per_hour(self) -> float:
        return float(self.event_count) / self.duration_hours

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": _iso(self.start),
            "end": _iso(self.end),
            "duration_hours": round(self.duration_hours, 4),
            "event_count": int(self.event_count),
            "defect_rate_per_hour": round(self.defect_rate_per_hour, 6),
        }


class OutcomeEvaluator:
    """Evaluate action outcomes by comparing before/after defect rates."""

    def __init__(
        self,
        events_db_path: str = DEFAULT_EVENTS_DB,
        actions_db_path: str = DEFAULT_ACTIONS_DB,
    ):
        self.events_db_path = events_db_path
        self.actions_db_path = actions_db_path

    def _get_action(self, action_id: str) -> dict[str, Any]:
        db_path = Path(self.actions_db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Actions DB not found: {db_path}")

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT action_id, timestamp, rca_reference, assigned_role, assigned_to,
                       status, acknowledged_at, resolved_by, approval_required, approved_by,
                       closure_notes, version, action_payload, updated_at
                FROM tracked_actions
                WHERE action_id = ?
                """,
                (action_id,),
            ).fetchone()

        if row is None:
            raise KeyError(f"Action '{action_id}' not found in tracker")

        return {
            "action_id": row["action_id"],
            "timestamp": row["timestamp"],
            "rca_reference": row["rca_reference"],
            "assigned_role": row["assigned_role"],
            "assigned_to": row["assigned_to"],
            "status": row["status"],
            "acknowledged_at": row["acknowledged_at"],
            "resolved_by": row["resolved_by"],
            "approval_required": bool(row["approval_required"]),
            "approved_by": row["approved_by"],
            "closure_notes": row["closure_notes"],
            "version": int(row["version"]),
            "action_payload": json.loads(row["action_payload"] or "{}"),
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _classify_6m_for_outcome(action: dict[str, Any]) -> str:
        """Deterministic 6M mapping for outcome reports."""
        payload = action.get("action_payload") or {}

        explicit = str(payload.get("classification_6m", "") or "").strip()
        if explicit in VALID_6M:
            return explicit

        target = str(payload.get("target_system", "") or "").lower()
        if target == "operator":
            return "Man"
        if target == "machine":
            return "Machine"

        root_factor = str(payload.get("root_cause_factor", "") or "").lower()
        explanation = str(payload.get("explanation", "") or "").lower()
        merged = f"{root_factor} {explanation}"

        if "operator" in merged:
            return "Man"
        if "machine" in merged:
            return "Machine"
        if "shift" in merged or "environment" in merged:
            return "Environment"
        if "defect_type" in merged or "material" in merged:
            return "Material"
        if "zscore" in merged or "anomaly" in merged or "sensor" in merged or "measurement" in merged:
            return "Measurement"
        if "trend" in merged or "sequence" in merged or "process" in merged or "method" in merged:
            return "Method"

        # Deterministic default if no evidence terms are present.
        return "Method"

    def _count_events(self, start: datetime, end: datetime) -> int:
        db_path = Path(self.events_db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Events DB not found: {db_path}")

        with sqlite3.connect(str(db_path)) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS c
                FROM defect_events
                WHERE julianday(timestamp) >= julianday(?) AND julianday(timestamp) < julianday(?)
                """,
                (_iso(start), _iso(end)),
            ).fetchone()
        return int(row[0] if row else 0)

    @staticmethod
    def _rate_significance(before: WindowStats, after: WindowStats) -> dict[str, Any]:
        """
        Two-sample rate comparison (normal approximation, pooled-rate z-test).
        H0: lambda_before == lambda_after
        """
        c1, t1 = float(before.event_count), before.duration_hours
        c2, t2 = float(after.event_count), after.duration_hours

        r1 = c1 / t1
        r2 = c2 / t2
        pooled = (c1 + c2) / (t1 + t2)
        se = math.sqrt(max(pooled * (1.0 / t1 + 1.0 / t2), 1e-12))
        z = (r2 - r1) / se
        p = 2.0 * (1.0 - _norm_cdf(abs(z)))

        return {
            "method": "pooled_rate_z_test",
            "z_score": round(float(z), 6),
            "p_value": round(float(p), 8),
            "significant_at_0_05": bool(p < 0.05),
        }

    def evaluate_action_outcome(
        self,
        action_id: str,
        *,
        before_hours: float = 2.0,
        after_hours: float = 2.0,
        production_volume: float | dict[str, float] | None = None,
        scrap_rate: float | dict[str, float] | None = None,
        cycle_time: float | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Compare defect rate before and after an action and compute improvement/significance.
        """
        action = self._get_action(action_id)

        action_start = _parse_ts(action.get("acknowledged_at")) or _parse_ts(action.get("timestamp"))
        if not action_start:
            raise ValueError(f"Action '{action_id}' has no valid start timestamp")

        # For after-window anchor, use closure/update time (if closed), else current UTC.
        action_end = _parse_ts(action.get("updated_at")) or datetime.now(timezone.utc)

        before_end = action_start
        before_start = before_end - timedelta(hours=float(before_hours))

        after_start = action_end
        after_end = after_start + timedelta(hours=float(after_hours))

        before_count = self._count_events(before_start, before_end)
        after_count = self._count_events(after_start, after_end)

        before_stats = WindowStats(start=before_start, end=before_end, event_count=before_count)
        after_stats = WindowStats(start=after_start, end=after_end, event_count=after_count)

        before_rate = before_stats.defect_rate_per_hour
        after_rate = after_stats.defect_rate_per_hour
        improvement_pct = ((before_rate - after_rate) / before_rate * 100.0) if before_rate > 0 else 0.0

        significance = self._rate_significance(before_stats, after_stats)
        classification_6m = self._classify_6m_for_outcome(action)

        def _pick_window_value(v: float | dict[str, float] | None, name: str) -> float | None:
            if v is None:
                return None
            if isinstance(v, dict):
                # accepted keys: before/after/default
                return float(v.get(name, v.get("default"))) if (v.get(name, v.get("default")) is not None) else None
            return float(v)

        prod_before = _pick_window_value(production_volume, "before")
        prod_after = _pick_window_value(production_volume, "after")
        scrap_before = _pick_window_value(scrap_rate, "before")
        scrap_after = _pick_window_value(scrap_rate, "after")
        cycle_before = _pick_window_value(cycle_time, "before")
        cycle_after = _pick_window_value(cycle_time, "after")

        def _safe_impr(before: float | None, after: float | None, *, higher_is_better: bool = False) -> float | None:
            if before is None or after is None or abs(before) <= 1e-12:
                return None
            delta = (after - before) if higher_is_better else (before - after)
            return (delta / before) * 100.0

        kpi_metrics = {
            "production_volume": {
                "before": prod_before,
                "after": prod_after,
                # More output is an improvement; for scrap and cycle time lower is better.
                "improvement_percent": _safe_impr(prod_before, prod_after, higher_is_better=True),
            },
            "scrap_rate": {
                "before": scrap_before,
                "after": scrap_after,
                "improvement_percent": _safe_impr(scrap_before, scrap_after),
            },
            "cycle_time": {
                "before": cycle_before,
                "after": cycle_after,
                "improvement_percent": _safe_impr(cycle_before, cycle_after),
            },
        }

        # Production-aligned combined KPI signal (optional, deterministic)
        # Higher signal means stronger operational improvement.
        kpi_components: list[float] = []
        vol_imp = kpi_metrics["production_volume"]["improvement_percent"]
        if vol_imp is not None:
            kpi_components.append(max(-100.0, min(100.0, float(vol_imp))) / 100.0)
        scrap_imp = kpi_metrics["scrap_rate"]["improvement_percent"]
        if scrap_imp is not None:
            kpi_components.append(max(-100.0, min(100.0, float(scrap_imp))) / 100.0)
        cycle_imp = kpi_metrics["cycle_time"]["improvement_percent"]
        if cycle_imp is not None:
            kpi_components.append(max(-100.0, min(100.0, float(cycle_imp))) / 100.0)

        kpi_effect_signal = (sum(kpi_components) / len(kpi_components)) if kpi_components else None

        effective_with_kpi = bool(
            improvement_pct > 0
            and significance["significant_at_0_05"]
            and (kpi_effect_signal is None or kpi_effect_signal >= 0.0)
        )

        return {
            "action_reference": {
                "action_id": action["action_id"],
                "rca_reference": action["rca_reference"],
                "classification_6m": classification_6m,
                "status": action["status"],
                "assigned_role": action["assigned_role"],
                "assigned_to": action["assigned_to"],
                "resolved_by": action["resolved_by"],
                "approval_required": action["approval_required"],
                "approved_by": action["approved_by"],
            },
            "comparison_windows": {
                "before_action": before_stats.to_dict(),
                "after_action": after_stats.to_dict(),
            },
            "outcome_metrics": {
                "before_rate_per_hour": round(before_rate, 6),
                "after_rate_per_hour": round(after_rate, 6),
                "improvement_percent": round(float(improvement_pct), 4),
                "kpi_metrics": {
                    "production_volume": {
                        "before": kpi_metrics["production_volume"]["before"],
                        "after": kpi_metrics["production_volume"]["after"],
                        "improvement_percent": (
                            None
                            if kpi_metrics["production_volume"]["improvement_percent"] is None
                            else round(float(kpi_metrics["production_volume"]["improvement_percent"]), 4)
                        ),
                    },
                    "scrap_rate": {
                        "before": kpi_metrics["scrap_rate"]["before"],
                        "after": kpi_metrics["scrap_rate"]["after"],
                        "improvement_percent": (
                            None
                            if kpi_metrics["scrap_rate"]["improvement_percent"] is None
                            else round(float(kpi_metrics["scrap_rate"]["improvement_percent"]), 4)
                        ),
                    },
                    "cycle_time": {
                        "before": kpi_metrics["cycle_time"]["before"],
                        "after": kpi_metrics["cycle_time"]["after"],
                        "improvement_percent": (
                            None
                            if kpi_metrics["cycle_time"]["improvement_percent"] is None
                            else round(float(kpi_metrics["cycle_time"]["improvement_percent"]), 4)
                        ),
                    },
                    "kpi_effect_signal": None if kpi_effect_signal is None else round(float(kpi_effect_signal), 4),
                },
            },
            "statistical_significance": significance,
            "outcome_summary": {
                "effective": effective_with_kpi,
                "direction": "improved" if improvement_pct > 0 else "worsened_or_flat",
                "narrative": (
                    f"Defect rate changed from {before_rate:.3f}/h to {after_rate:.3f}/h "
                    f"({improvement_pct:.2f}% improvement); p={significance['p_value']:.6f}."
                ),
            },
            "audit": {
                "generated_at_utc": _iso(datetime.now(timezone.utc)),
                "before_hours": float(before_hours),
                "after_hours": float(after_hours),
                "kpi_inputs_used": {
                    "production_volume": production_volume,
                    "scrap_rate": scrap_rate,
                    "cycle_time": cycle_time,
                },
                "events_db_path": self.events_db_path,
                "actions_db_path": self.actions_db_path,
                "schema_version": "1.0.0",
            },
        }

    def evaluate_recent_closed_actions(
        self,
        *,
        limit: int = 20,
        before_hours: float = 2.0,
        after_hours: float = 2.0,
        production_volume: float | dict[str, float] | None = None,
        scrap_rate: float | dict[str, float] | None = None,
        cycle_time: float | dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Batch evaluate recent closed actions from tracker DB."""
        db_path = Path(self.actions_db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Actions DB not found: {db_path}")

        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                """
                SELECT action_id
                FROM tracked_actions
                WHERE status = 'closed'
                ORDER BY datetime(updated_at) DESC
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()

        reports = []
        for r in rows:
            aid = str(r[0])
            try:
                reports.append(
                    self.evaluate_action_outcome(
                        aid,
                        before_hours=before_hours,
                        after_hours=after_hours,
                        production_volume=production_volume,
                        scrap_rate=scrap_rate,
                        cycle_time=cycle_time,
                    )
                )
            except Exception as exc:
                reports.append({"action_id": aid, "error": str(exc)})

        return {
            "reports": reports,
            "summary": {
                "evaluated": len([r for r in reports if "error" not in r]),
                "errors": len([r for r in reports if "error" in r]),
                "effective_count": len(
                    [
                        r
                        for r in reports
                        if isinstance(r, dict)
                        and r.get("outcome_summary", {}).get("effective") is True
                    ]
                ),
            },
        }


if __name__ == "__main__":
    evaluator = OutcomeEvaluator()
    batch = evaluator.evaluate_recent_closed_actions(limit=5, before_hours=1.0, after_hours=1.0)
    print(json.dumps(batch, indent=2))