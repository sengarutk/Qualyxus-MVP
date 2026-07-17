from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.analytics.action_engine import build_actions
from src.analytics.outcome_evaluator import OutcomeEvaluator
from src.analytics.root_cause import run_root_cause


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RCAReportInputs:
    rca_output: dict[str, Any]
    actions: dict[str, Any]
    outcome_evaluation: dict[str, Any]


class RCAReportGenerator:
    """Industry-grade structured RCA report generator (JSON + formatted text)."""

    @staticmethod
    def _window_snapshot(rca_output: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        for k in ["last_hour", "last_shift", "last_24h", "all_time"]:
            if isinstance(rca_output.get(k), dict):
                return k, rca_output[k]
        return "current", rca_output

    @staticmethod
    def _problem_statement(rca_output: dict[str, Any]) -> dict[str, Any]:
        win_name, win = RCAReportGenerator._window_snapshot(rca_output)
        event_count = int(win.get("event_count", 0) or 0)
        anomalies = (win.get("anomalies", {}) or {}).get("spikes", []) or []
        confidence = float((win.get("confidence", {}) or {}).get("overall_confidence", 0.0) or 0.0)

        return {
            "window": win_name,
            "event_count": event_count,
            "anomaly_count": len(anomalies),
            "overall_confidence": round(confidence, 4),
            "statement": (
                f"Elevated defect condition detected in {win_name}: "
                f"{event_count} events with {len(anomalies)} anomaly spikes "
                f"(confidence={confidence:.2f})."
            ),
        }

    @staticmethod
    def _root_cause_summary(rca_output: dict[str, Any]) -> dict[str, Any]:
        win_name, win = RCAReportGenerator._window_snapshot(rca_output)
        explanations = win.get("explanations", []) or []
        classes = win.get("classification_6m", []) or []

        top_insights = [
            {
                "insight": e.get("insight"),
                "classification_6m": e.get("classification_6m"),
                "evidence_summary": e.get("evidence_summary", {}),
            }
            for e in explanations[:5]
        ]

        return {
            "window": win_name,
            "classifications_6m": classes,
            "top_insights": top_insights,
            "summary_text": (
                "RCA indicates dominant contributors mapped to 6M classes with deterministic evidence "
                "from correlation, anomaly, and trend signals."
            ),
        }

    @staticmethod
    def _actions_section(actions: dict[str, Any], key: str) -> list[dict[str, Any]]:
        out = []
        for a in (actions.get(key, []) or [])[:10]:
            out.append(
                {
                    "title": a.get("title"),
                    "classification_6m": a.get("classification_6m"),
                    "urgency_level": a.get("urgency_level"),
                    "responsible_role": a.get("responsible_role"),
                    "expected_resolution_time": a.get("expected_resolution_time"),
                    "severity_score": a.get("severity_score"),
                    "explanation": a.get("explanation"),
                }
            )
        return out

    @staticmethod
    def _extract_outcome_report(outcome_evaluation: dict[str, Any]) -> dict[str, Any]:
        if isinstance(outcome_evaluation.get("action_reference"), dict):
            return outcome_evaluation
        reports = outcome_evaluation.get("reports", []) or []
        for r in reports:
            if isinstance(r, dict) and "error" not in r:
                return r
        return {}

    @staticmethod
    def _verification_plan(actions: dict[str, Any], outcome_evaluation: dict[str, Any]) -> dict[str, Any]:
        report = RCAReportGenerator._extract_outcome_report(outcome_evaluation)
        metrics = report.get("outcome_metrics", {}) if isinstance(report, dict) else {}
        sig = report.get("statistical_significance", {}) if isinstance(report, dict) else {}

        priority = actions.get("priority_actions", []) or []
        checkpoints = []
        for idx, a in enumerate(priority[:3], start=1):
            checkpoints.append(
                {
                    "step": idx,
                    "check": f"Validate '{a.get('title', 'action')}' effectiveness in next monitoring window",
                    "owner": a.get("responsible_role", "QA"),
                    "due": a.get("expected_resolution_time", "< 24 h"),
                }
            )

        return {
            "defect_rate_before": metrics.get("before_rate_per_hour"),
            "defect_rate_after": metrics.get("after_rate_per_hour"),
            "improvement_percent": metrics.get("improvement_percent"),
            "statistical_significance": sig,
            "checkpoints": checkpoints,
            "acceptance_criteria": {
                "minimum_improvement_percent": 5.0,
                "p_value_threshold": 0.05,
                "required": "Improvement >= 5% and p-value < 0.05",
            },
        }

    @staticmethod
    def _kpi_impact_summary(outcome_evaluation: dict[str, Any]) -> dict[str, Any]:
        report = RCAReportGenerator._extract_outcome_report(outcome_evaluation)
        metrics = report.get("outcome_metrics", {}) if isinstance(report, dict) else {}
        kpi = (metrics.get("kpi_metrics", {}) or {}) if isinstance(metrics, dict) else {}

        return {
            "defect_rate_improvement_percent": metrics.get("improvement_percent"),
            "kpi_metrics": kpi,
            "summary_text": (
                "KPI impact summarizes defect-rate delta and optional production KPIs "
                "(production volume, scrap rate, cycle time)."
            ),
        }

    def generate_report(
        self,
        *,
        rca_output: dict[str, Any],
        actions: dict[str, Any],
        outcome_evaluation: dict[str, Any],
        report_id: str | None = None,
    ) -> dict[str, Any]:
        """Generate structured RCA report JSON."""
        report_id = report_id or f"rca-report-{uuid.uuid4().hex[:12]}"

        return {
            "report_id": report_id,
            "generated_at_utc": _utc_now(),
            "problem_statement": self._problem_statement(rca_output),
            "containment_actions": self._actions_section(actions, "containment_actions"),
            "root_cause_summary": self._root_cause_summary(rca_output),
            "corrective_actions": self._actions_section(actions, "corrective_actions"),
            "preventive_actions": self._actions_section(actions, "preventive_actions"),
            "verification_plan": self._verification_plan(actions, outcome_evaluation),
            "kpi_impact_summary": self._kpi_impact_summary(outcome_evaluation),
            "audit": {
                "schema_version": "1.0.0",
                "source_sections": ["rca_output", "actions", "outcome_evaluation"],
            },
        }

    @staticmethod
    def format_report_text(report_json: dict[str, Any]) -> str:
        """Generate formatted text version of structured RCA report."""
        ps = report_json.get("problem_statement", {})
        rcs = report_json.get("root_cause_summary", {})
        vp = report_json.get("verification_plan", {})
        kpi = report_json.get("kpi_impact_summary", {})

        def _fmt_actions(title: str, items: list[dict[str, Any]]) -> str:
            if not items:
                return f"{title}:\n  - None\n"
            lines = [f"{title}:"]
            for a in items:
                lines.append(
                    "  - "
                    f"{a.get('title')} | 6M={a.get('classification_6m')} | "
                    f"Urgency={a.get('urgency_level')} | Owner={a.get('responsible_role')} | "
                    f"ETA={a.get('expected_resolution_time')}"
                )
            return "\n".join(lines) + "\n"

        text = []
        text.append("=" * 88)
        text.append("INDUSTRIAL RCA REPORT")
        text.append("=" * 88)
        text.append(f"Report ID: {report_json.get('report_id')}")
        text.append(f"Generated: {report_json.get('generated_at_utc')}")
        text.append("")
        text.append("1) PROBLEM STATEMENT")
        text.append(f"- Window: {ps.get('window')}")
        text.append(f"- Event Count: {ps.get('event_count')}")
        text.append(f"- Anomaly Count: {ps.get('anomaly_count')}")
        text.append(f"- Confidence: {ps.get('overall_confidence')}")
        text.append(f"- Statement: {ps.get('statement')}")
        text.append("")

        text.append(_fmt_actions("2) CONTAINMENT ACTIONS", report_json.get("containment_actions", [])))

        text.append("3) ROOT CAUSE SUMMARY")
        text.append(f"- Summary: {rcs.get('summary_text')}")
        classes = rcs.get("classifications_6m", []) or []
        if classes:
            text.append("- 6M Classifications:")
            for c in classes:
                text.append(f"  - {c.get('classification_6m')}: {c.get('basis')}")
        insights = rcs.get("top_insights", []) or []
        if insights:
            text.append("- Top Insights:")
            for i in insights:
                text.append(f"  - {i.get('insight')} (6M={i.get('classification_6m')})")
        text.append("")

        text.append(_fmt_actions("4) CORRECTIVE ACTIONS", report_json.get("corrective_actions", [])))
        text.append(_fmt_actions("5) PREVENTIVE ACTIONS", report_json.get("preventive_actions", [])))

        text.append("6) VERIFICATION PLAN")
        text.append(f"- Defect Rate Before: {vp.get('defect_rate_before')}")
        text.append(f"- Defect Rate After: {vp.get('defect_rate_after')}")
        text.append(f"- Improvement %: {vp.get('improvement_percent')}")
        sig = vp.get("statistical_significance", {}) or {}
        text.append(f"- Significance: p={sig.get('p_value')} (significant={sig.get('significant_at_0_05')})")
        checks = vp.get("checkpoints", []) or []
        if checks:
            text.append("- Checkpoints:")
            for c in checks:
                text.append(f"  - Step {c.get('step')}: {c.get('check')} | Owner={c.get('owner')} | Due={c.get('due')}")
        text.append("")

        text.append("7) KPI IMPACT SUMMARY")
        text.append(f"- Defect Rate Improvement %: {kpi.get('defect_rate_improvement_percent')}")
        km = kpi.get("kpi_metrics", {}) or {}
        if km:
            for key in ["production_volume", "scrap_rate", "cycle_time"]:
                row = km.get(key, {}) or {}
                text.append(
                    f"- {key}: before={row.get('before')}, after={row.get('after')}, "
                    f"improvement%={row.get('improvement_percent')}"
                )
            text.append(f"- KPI Effect Signal: {km.get('kpi_effect_signal')}")
        text.append(f"- Summary: {kpi.get('summary_text')}")
        text.append("")
        text.append("=" * 88)

        return "\n".join(text)

    def generate_report_package(
        self,
        *,
        rca_output: dict[str, Any],
        actions: dict[str, Any],
        outcome_evaluation: dict[str, Any],
        report_id: str | None = None,
    ) -> dict[str, Any]:
        report_json = self.generate_report(
            rca_output=rca_output,
            actions=actions,
            outcome_evaluation=outcome_evaluation,
            report_id=report_id,
        )
        return {
            "report_json": report_json,
            "report_text": self.format_report_text(report_json),
        }


if __name__ == "__main__":
    rca = run_root_cause()
    actions = build_actions(rca)
    evaluator = OutcomeEvaluator()
    outcomes = evaluator.evaluate_recent_closed_actions(limit=1, before_hours=1.0, after_hours=1.0)

    generator = RCAReportGenerator()
    pkg = generator.generate_report_package(
        rca_output=rca,
        actions=actions,
        outcome_evaluation=outcomes,
    )

    print(json.dumps(pkg, indent=2))