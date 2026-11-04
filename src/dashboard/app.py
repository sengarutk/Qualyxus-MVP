from __future__ import annotations

import os
import time
import json
import logging
from typing import Any
from pathlib import Path
import plotly.express as px
import streamlit as st

logger = logging.getLogger("QualyxusDashboard")

def _safe_parse_telemetry_float(value: Any, default: float = 0.0) -> float:
    """
    Defensive Type-Casting Filter.
    Guarantees that unparseable alpha strings, null references, or malformed
    telemetry elements cannot trigger unhandled ValueError exceptions.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        # Gracefully log data anomalies while maintaining user interface uptime
        logger.debug("[Dashboard] Non-numeric telemetry value encountered: %s", value)
        return default


from src.analytics.insight_generator import generate_insights
from src.analytics.root_cause import run_root_cause
from src.config.config import Config
from src.dashboard.components import kpi_row, show_events_table
from src.operations.action_tracker import ActionTracker
from src.operations.alert_manager import build_alerts, load_unresolved_actions
from src.operations.rbac import (
    Role,
    Permission,
    can_close_actions,
    can_approve_actions,
    can_view_rca,
    can_view_reports,
    enforce_permission,
)
from src.operations.shift_handover import ShiftHandoverGenerator, ShiftHandoverInputs
from src.dashboard.data_access import (
    fetch_esg_data,
    fetch_federated_status,
    fetch_process_overview,
    fetch_vision_overview,
    fetch_system_health,
    get_kpi_metrics,
    query_defects_by_machine,
    query_defects_by_shift,
    query_defects_per_hour,
    query_hourly_heatmap,
    query_live_events,
    query_severity_distribution,
    events_table_ready,
    query_timeline_events,
)


st.set_page_config(page_title="Qualyxus Dashboard", layout="wide")


def _fmt_uptime(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


@st.cache_data(ttl=10)
def _load_rca(sqlite_path: str) -> dict:
    """Load RCA payload with lightweight cache for fast dashboard refresh."""
    return run_root_cause(sqlite_path=sqlite_path)


@st.cache_data(ttl=5)
def _load_active_alerts(rca_payload: dict, actions_db_path: str) -> dict:
    unresolved = load_unresolved_actions(sqlite_path=actions_db_path)
    return build_alerts(rca_payload, unresolved_actions=unresolved)


@st.cache_data(ttl=5)
def _load_open_actions(actions_db_path: str, limit: int = 30) -> list[dict]:
    tracker = ActionTracker(sqlite_path=actions_db_path)
    open_rows = tracker.list_actions(status="open", limit=limit)
    in_progress_rows = tracker.list_actions(status="in_progress", limit=limit)
    return (open_rows + in_progress_rows)[:limit]


@st.cache_data(ttl=10)
def _load_shift_handover(events_db_path: str, actions_db_path: str) -> dict:
    gen = ShiftHandoverGenerator(
        config=ShiftHandoverInputs(events_db_path=events_db_path, actions_db_path=actions_db_path)
    )
    return gen.generate_handover()


@st.cache_data(ttl=5)
def _load_runtime_metrics(metrics_path: str) -> dict:
    p = Path(metrics_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _window_confidence(window_payload: dict) -> float:
    return float((window_payload.get("confidence", {}) or {}).get("overall_confidence", 0.0) or 0.0)


def _window_top_insights(window_payload: dict, top_n: int = 3) -> list[dict]:
    insights = generate_insights(window_payload)
    return insights[:top_n]


def _verification_badge(action: dict) -> str:
    status = str(action.get("verification_status", "PENDING") or "PENDING").upper()
    mapping = {
        "PENDING": "🔵 Pending",
        "VERIFIED": "✅ Verified",
        "FAILED": "❌ Failed",
        "TIMEOUT": "❌ Timeout",
    }
    return mapping.get(status, status.title())


def main():
    cfg = Config.load()
    sqlite_path = cfg.get("logging", "sqlite_path", default="data/events/events.db")
    actions_db_path = str(Path(sqlite_path).parent / "actions.db")
    status_path = os.getenv(
        "CAMERA_STATUS_PATH",
        str(cfg.get("camera_manager", "status_path", default="data/live/camera_status.json")),
    )
    runtime_metrics_path = str(cfg.get("runtime_metrics", "output_path", default="data/runtime/runtime_metrics.json"))
    systemic_insight_path = Path(cfg.get("paths", "data_dir", default="data")) / "runtime" / "systemic_insight.json"

    st.title("🏭 Qualyxus Operational KPI Dashboard")
    auto_refresh = st.sidebar.toggle("Auto-refresh every 5s", value=True)
    selected_role = st.sidebar.selectbox(
        "Current role",
        options=[r.value for r in Role],
        index=3,  # manager by default
        help="Role-based visibility is applied in Operations Control Panel.",
    )
    current_user = st.sidebar.text_input(
        "Current user",
        value=f"{selected_role}:user-01",
        help="Used for action workflow audit fields (acknowledge/approve/close).",
    )

    system_health = fetch_system_health()
    state = str(system_health.get("state", "SAFE") or "SAFE")
    badge = {
        "SAFE": "🟢 Optimal",
        "WARNING": "🟡 Throttled",
        "CRITICAL": "🔴 Critical",
        "EMERGENCY": "🚨 Emergency",
    }.get(state, state)
    vitals = system_health.get("vitals", {}) or {}
    st.sidebar.markdown("## System Vitals")
    st.sidebar.metric("Guardian State", badge)
    
    # Cleanly filter telemetry items via defensive parsing wrapper
    vram_pct = _safe_parse_telemetry_float(vitals.get("vram_percent", 0.0))
    cpu_pct = _safe_parse_telemetry_float(vitals.get("cpu_load_percent", 0.0))
    cpu_temp = _safe_parse_telemetry_float(vitals.get("cpu_temperature_c", 0.0))
    gpu_temp = _safe_parse_telemetry_float(vitals.get("gpu_temperature_c", 0.0))
    
    # Flaw Fix (Issue 6): Clamp floats strictly between 0.0 and 1.0 to prevent Streamlit rendering exceptions 
    vram_clamped = max(0.0, min(1.0, vram_pct / 100.0))
    cpu_clamped = max(0.0, min(1.0, cpu_pct / 100.0))
    
    st.sidebar.progress(vram_clamped, text=f"VRAM {vram_pct:.1f}%")
    st.sidebar.progress(cpu_clamped, text=f"CPU Load {cpu_pct:.1f}%")
    st.sidebar.caption(f"CPU Temp: {cpu_temp:.1f}°C | GPU Temp: {gpu_temp:.1f}°C")
    with st.sidebar.expander("Resource Log"):
        for item in list(system_health.get("resource_log", []) or [])[:20]:
            st.write(f"- {item}")

    if not events_table_ready(sqlite_path):
        st.info("No inspection data yet. Start the edge engine, then reload this page.")
        st.code("python main.py", language="bash")
        st.stop()

    timeline_df = query_timeline_events(sqlite_path=sqlite_path, limit=300)
    if timeline_df.empty:
        st.success("No defects recorded yet.")

    kpis = get_kpi_metrics(sqlite_path=sqlite_path, status_path=status_path)
    kpi_row(
        defects_per_hour=kpis["defects_per_hour"],
        avg_confidence=kpis["avg_confidence"],
        top_defect_type=kpis["top_defect_type"],
        active_cameras=kpis["active_cameras"],
        uptime_text=_fmt_uptime(kpis["system_uptime_seconds"]),
    )

    if systemic_insight_path.exists():
        try:
            systemic_payload = json.loads(systemic_insight_path.read_text(encoding="utf-8"))
            systemic_text = str(systemic_payload.get("insight", "") or "").strip()
            if systemic_text:
                st.markdown("### 🧠 Global Strategic Intelligence")
                st.info(systemic_text)
        except Exception:
            pass

    if bool(cfg.get("inspection", "enabled", default=True)):
        _render_vision_section(cfg, status_path)

    if bool(cfg.get("process", "enabled", default=False)):
        _render_process_section(cfg, actions_db_path)

    if bool(cfg.get("esg_params", "enabled", default=True)):
        _render_esg_section(sqlite_path)

    if bool(cfg.get("federated_learning", "enabled", default=False)):
        _render_federated_section()

    _render_operations(cfg, sqlite_path, actions_db_path, status_path, runtime_metrics_path,
                       selected_role, current_user, kpis, auto_refresh)


def _render_vision_section(cfg, status_path: str) -> None:
    st.divider()
    st.subheader("🔬 Vision Models & Inspection Quality")
    required = tuple(str(c.get("id")) for c in (cfg.get("cameras", default=[]) or [])
                     if c.get("enabled", True) and c.get("required_view"))
    overview = fetch_vision_overview(
        status_path,
        str(cfg.get("process", "db_path", default="data/events/process.db")),
        cfg.get("inspection", "registry_dir", default=None),
        str(cfg.get("inspection", "review", "dir", default="data/review")),
        str(cfg.get("inspection", "audit", "dir", default="data/audit")),
        required_cameras=required,
    )
    parts = overview.get("parts") or {}
    review = overview.get("review") or {}
    audit = overview.get("audit")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Parts judged (8h)", parts.get("total", 0), help=f"{parts.get('multi_view', 0)} seen by several cameras")
    c2.metric("Failed parts", parts.get("fail", 0))
    c3.metric("Incomplete (missing view)", parts.get("incomplete", 0),
              help="Passed parts not yet seen by every camera marked required_view: true")
    agreement = review.get("model_agreement")
    agreement_text = "n/a" if agreement is None else f"{100 * agreement:.1f}%"
    c4.metric("Review queue", review.get("pending", 0),
              help=f"{review.get('reviewed', 0)} reviewed; model agreed with reviewers {agreement_text}")
    if audit is None:
        enabled = bool(cfg.get("inspection", "audit", "enabled", default=True))
        c5.metric("Audit trail", "no records yet" if enabled else "off")
    else:
        c5.metric("Audit trail", "✅ intact" if audit.get("ok") else "❌ BROKEN", help=str(audit.get("error") or
                  f"{audit.get('records', 0)} hash-chained verdict records"))

    if overview.get("recent_parts"):
        st.markdown("**Latest parts, fused across every camera view**")
        st.dataframe(overview["recent_parts"], width="stretch", hide_index=True)

    if overview.get("cameras"):
        st.markdown("**Model health per camera**")
        st.dataframe(overview["cameras"], width="stretch", hide_index=True)
        for cam in overview["cameras"]:
            if cam.get("alerts"):
                st.warning(f"{cam['camera']}: {cam['alerts']}")
    left, right = st.columns(2)
    with left:
        if overview.get("defect_types"):
            fig = px.bar(overview["defect_types"], x="defect_type", y="parts", title="Failed parts by defect type (8h)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("No failed parts in the window.")
    with right:
        st.markdown("**Most informative parts to review next**")
        if review.get("next"):
            st.dataframe(review["next"], width="stretch", hide_index=True)
            st.caption("Export a labelling batch: python -m src.inference.labeling export --out data/labeling/batch_N")
        else:
            st.caption("Review queue is empty.")
    if overview.get("registry"):
        st.markdown("**Model registry**")
        st.dataframe(overview["registry"], width="stretch", hide_index=True)
    for rep in overview.get("shadow") or []:
        if not rep.get("parts"):
            continue
        reviewed = rep.get("reviewed") or {}
        st.info(
            f"Shadow {rep['shadow_version']} vs active {rep['active_version']} ({rep['recipe']}): "
            f"{rep['parts']} parts, agreement {100 * rep['agreement']:.1f}%, reject rate "
            f"{100 * rep['shadow_reject_rate']:.1f}% vs {100 * rep['active_reject_rate']:.1f}%"
            + (f"; on {reviewed['parts']} reviewed parts accuracy {100 * reviewed['shadow_accuracy']:.1f}% vs "
               f"{100 * reviewed['active_accuracy']:.1f}%, escapes {reviewed['shadow_escapes']} vs "
               f"{reviewed['active_escapes']}" if reviewed else "")
        )
    if overview.get("requests"):
        st.markdown("**Retraining requests (need approval)**")
        st.dataframe(overview["requests"], width="stretch", hide_index=True)
        st.caption("Approve: python -m src.inference.model_registry approve <id> --by <name>, then retrain <id>.")


def _render_process_section(cfg, actions_db_path: str) -> None:
    st.divider()
    st.subheader("⚙️ Process ↔ Quality Intelligence")
    overview = fetch_process_overview(
        str(cfg.get("process", "db_path", default="data/events/process.db")), actions_db_path,
    )
    if not overview:
        st.info("No process cycles recorded yet. Start the runtime with process.enabled: true.")
        return
    k = overview["kpis"]
    model = overview.get("model") or {}
    validation = (model.get("metrics") or {}).get("validation") or {}
    prediction = k.get("prediction") or {}

    def _pct(value):
        return "n/a" if value is None else f"{100 * value:.1f}%"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Parts inspected (8h)", k.get("parts_inspected", 0))
    c2.metric("Scrap rate", _pct(k.get("scrap_rate")))
    c3.metric("Risk model", f"{model.get('kind', 'none')}", help=str(model.get("version", "")))
    c4.metric("Validation AUC", f"{validation['roc_auc']:.2f}" if "roc_auc" in validation else "n/a")
    c5.metric("Median early warning", "n/a" if k.get("median_warning_cycles") is None
              else f"{k['median_warning_cycles']:.0f} cycles")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Risk precision", _pct(prediction.get("precision")))
    d2.metric("Risk recall", _pct(prediction.get("recall")))
    d3.metric("False reject rate", _pct(k.get("false_reject_rate")), help="Requires audit records")
    d4.metric("Escape rate", _pct(k.get("escape_rate")), help="Requires audit records")

    timeline = overview.get("timeline")
    if timeline is not None and not timeline.empty:
        timeline = timeline.assign(outcome=timeline["failed"].map({1: "failed inspection", 0: "passed"}).fillna("not inspected"))
        fig = px.scatter(
            timeline, x="time", y="risk", color="outcome",
            color_discrete_map={"failed inspection": "#d62728", "passed": "#2ca02c", "not inspected": "#9e9e9e"},
            hover_data=["cycle_id", "likely_defect", "factors"], title="Predicted defect risk per cycle vs. actual outcome",
        )
        st.plotly_chart(fig, width="stretch")
        high = timeline[timeline["decision"] == "high"].sort_values("ts", ascending=False).head(15)
        if not high.empty:
            st.markdown("**Recent high-risk cycles**")
            st.dataframe(high[["time", "cycle_id", "risk", "likely_defect", "factors", "failed"]],
                         width="stretch", hide_index=True)

    prescriptions = overview.get("prescriptions") or []
    st.markdown(f"**Setpoint prescriptions awaiting action ({len(prescriptions)})**")
    if prescriptions:
        st.caption("Approve and close them in the Operations Control Panel once the change is applied on the machine.")
        st.dataframe(prescriptions, width="stretch", hide_index=True)
    else:
        st.caption("No open prescriptions.")


def _render_esg_section(sqlite_path: str) -> None:
    esg_payload = fetch_esg_data(sqlite_path=sqlite_path)
    esg_current = esg_payload.get("current", {})
    esg_delta = esg_payload.get("delta", {})
    currency_symbol = esg_current.get("currency_symbol", "$")

    st.divider()
    st.subheader("🌱 Financial & Environmental Impact")
    e1, e2, e3 = st.columns(3)
    e1.metric(
        "Scrap Prevented (kg)",
        f"{float(esg_current.get('scrap_prevented_kg', 0.0)):.2f}",
        delta=f"{float(esg_delta.get('scrap_prevented_kg', 0.0)):.2f} vs prev 24h",
    )
    e2.metric(
        f"Revenue Recovered ({currency_symbol})",
        f"{float(esg_current.get('revenue_recovered_usd', 0.0)):.2f}",
        delta=f"{float(esg_delta.get('revenue_recovered_usd', 0.0)):.2f} vs prev 24h",
    )
    e3.metric(
        "CO2 Emissions Offset (kg)",
        f"{float(esg_current.get('co2_emissions_offset_kg', 0.0)):.2f}",
        delta=f"{float(esg_delta.get('co2_emissions_offset_kg', 0.0)):.2f} vs prev 24h",
    )



def _render_federated_section() -> None:
    fed_status = fetch_federated_status()
    st.divider()
    st.subheader("🌐 Global Intelligence Status")
    f1, f2 = st.columns(2)
    f1.metric("Collaborative Contribution Count", int(fed_status.get("collaborative_contribution_count", 0) or 0))
    f2.metric("Global Knowledge Version", str(fed_status.get("global_knowledge_version", "n/a")))
    st.caption(str(fed_status.get("privacy_tooltip", "Qualyxus Federated Learning keeps raw data local.")))


def _render_operations(cfg, sqlite_path, actions_db_path, status_path, runtime_metrics_path,
                       selected_role, current_user, kpis, auto_refresh) -> None:
    try:
        rca = _load_rca(sqlite_path=sqlite_path)
    except Exception as exc:
        st.error(f"RCA engine unavailable: {exc}")
        rca = {}

    st.divider()
    st.subheader("🛠️ Operations Control Panel")

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**1) Active Alerts**")
        if can_view_reports(selected_role):
            alerts_payload = _load_active_alerts(rca_payload=rca, actions_db_path=actions_db_path)
            alerts = alerts_payload.get("alerts", []) or []
            st.metric("Active alerts", len(alerts))
            if alerts:
                st.dataframe(alerts[:20], width="stretch", hide_index=True)
            else:
                st.caption("No active alerts.")
        else:
            st.info("Not authorized for alerts panel (requires report visibility).")

    with p2:
        st.markdown("**2) Open Actions**")
        if can_close_actions(selected_role) or can_view_reports(selected_role):
            open_actions = _load_open_actions(actions_db_path=actions_db_path, limit=40)
            st.metric("Open/In-progress actions", len(open_actions))
            if open_actions:
                table_rows = [
                    {
                        "action_id": a.get("action_id"),
                        "status": a.get("status"),
                        "verification_badge": _verification_badge(a),
                        "assigned_role": a.get("assigned_role"),
                        "assigned_to": a.get("assigned_to"),
                        "rca_reference": a.get("rca_reference"),
                        "updated_at": a.get("updated_at"),
                    }
                    for a in open_actions
                ]
                st.dataframe(table_rows, width="stretch", hide_index=True)

                st.markdown("**Action Workflow**")
                action_options = [str(a.get("action_id")) for a in open_actions if a.get("action_id")]
                selected_action_id = st.selectbox(
                    "Select action",
                    options=action_options,
                    key="ops_action_workflow_select",
                )
                selected_action = next(
                    (a for a in open_actions if str(a.get("action_id")) == str(selected_action_id)),
                    {},
                )
                st.caption(
                    f"Status: {selected_action.get('status', 'n/a')} | "
                    f"Verification: {_verification_badge(selected_action)} | "
                    f"Approval required: {bool(selected_action.get('approval_required', False))}"
                )

                w1, w2, w3 = st.columns(3)
                with w1:
                    if st.button("Acknowledge", width="stretch"):
                        try:
                            enforce_permission(selected_role, Permission.CLOSE_ACTIONS, context="acknowledge action")
                            tracker = ActionTracker(sqlite_path=actions_db_path)
                            tracker.acknowledge_action(selected_action_id, user=current_user)
                            _load_open_actions.clear()
                            _load_active_alerts.clear()
                            _load_shift_handover.clear()
                            st.success(f"Action {selected_action_id} acknowledged by {current_user}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Acknowledge failed: {exc}")

                with w2:
                    if st.button("Approve", width="stretch"):
                        try:
                            enforce_permission(selected_role, Permission.APPROVE_ACTIONS, context="approve action")
                            tracker = ActionTracker(sqlite_path=actions_db_path)
                            tracker.approve_action(selected_action_id, user=current_user)
                            _load_open_actions.clear()
                            _load_active_alerts.clear()
                            _load_shift_handover.clear()
                            st.success(f"Action {selected_action_id} approved by {current_user}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Approve failed: {exc}")

                with w3:
                    if st.button("Close", width="stretch"):
                        try:
                            enforce_permission(selected_role, Permission.CLOSE_ACTIONS, context="close action")
                            tracker = ActionTracker(sqlite_path=actions_db_path)
                            tracker.update_action(
                                selected_action_id,
                                status="closed",
                                resolved_by=current_user,
                                changed_by=current_user,
                                change_reason="dashboard_close",
                                closure_notes="Closed via Operations Control Panel.",
                            )
                            _load_open_actions.clear()
                            _load_active_alerts.clear()
                            _load_shift_handover.clear()
                            st.success(f"Action {selected_action_id} closed by {current_user}.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Close failed: {exc}")
            else:
                st.caption("No open/in-progress actions.")
        else:
            st.info("Not authorized for actions panel.")

    p3, p4 = st.columns(2)
    with p3:
        st.markdown("**3) Shift Handover Summary**")
        if can_view_reports(selected_role):
            handover = _load_shift_handover(events_db_path=sqlite_path, actions_db_path=actions_db_path)
            hw = handover.get("handover_window", {})
            st.caption(f"{hw.get('shift_start', 'n/a')} → {hw.get('shift_end', 'n/a')}")
            st.metric("Shift events", int((handover.get("last_shift_events", {}) or {}).get("event_count", 0) or 0))
            st.metric("Critical alerts", int((handover.get("critical_alerts", {}) or {}).get("count", 0) or 0))
            st.metric("Open actions", int((handover.get("open_actions", {}) or {}).get("count", 0) or 0))
            st.write(handover.get("narrative_summary", "No handover narrative available."))
        else:
            st.info("Not authorized for shift handover panel.")

    with p4:
        st.markdown("**4) Latest RCA Snapshot**")
        if can_view_rca(selected_role):
            last_hour = rca.get("last_hour", {}) if isinstance(rca, dict) else {}
            confidence = _window_confidence(last_hour) if last_hour else 0.0
            st.metric("Last hour confidence", f"{confidence:.2f}")
            st.metric("Last hour events", int(last_hour.get("event_count", 0) or 0) if last_hour else 0)
            explanations = (last_hour.get("explanations", []) or [])[:3] if last_hour else []
            if explanations:
                for idx, ex in enumerate(explanations, start=1):
                    st.markdown(f"{idx}. {ex.get('insight', 'n/a')}")
            else:
                st.caption("No RCA snapshot explanations available.")
        else:
            st.info("Not authorized for RCA snapshot panel.")

    st.divider()
    st.subheader("🩺 System Health Monitoring")
    metrics_payload = _load_runtime_metrics(runtime_metrics_path)
    cameras_payload = (metrics_payload.get("cameras", {}) or {}) if isinstance(metrics_payload, dict) else {}

    if not cameras_payload:
        st.info(f"Runtime metrics unavailable. Expected file: {runtime_metrics_path}")
    else:
        cam_rows = []
        for cam_id, info in cameras_payload.items():
            q = (info.get("queue_sizes", {}) or {}) if isinstance(info, dict) else {}
            cam_rows.append(
                {
                    "camera_id": str(cam_id),
                    "fps": round(float(info.get("fps", 0.0) or 0.0), 3),
                    "dropped_frames": int(info.get("dropped_frames", 0) or 0),
                    "frame_queue": int(q.get("frame", 0) or 0),
                    "event_queue": int(q.get("event", 0) or 0),
                    "reconnect_count": int(info.get("reconnect_count", 0) or 0),
                    "restarts": int(info.get("restarts", 0) or 0),
                }
            )

        total_active_cameras = len(cam_rows)
        avg_fps = (sum(r["fps"] for r in cam_rows) / total_active_cameras) if total_active_cameras else 0.0
        total_dropped = sum(r["dropped_frames"] for r in cam_rows)
        uptime_text = _fmt_uptime(kpis.get("system_uptime_seconds", 0.0))

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Active cameras", total_active_cameras)
        h2.metric("Avg FPS", f"{avg_fps:.2f}")
        h3.metric("Dropped frames", total_dropped)
        h4.metric("Uptime", uptime_text)

        st.dataframe(cam_rows, width="stretch", hide_index=True)

    st.divider()
    st.subheader("💾 Storage & Retention Status")

    from src.system.data_retention import DataRetentionManager

    try:
        retention_mgr = DataRetentionManager.default()
        status = retention_mgr.get_retention_status()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Files", status["total_files"])
        col2.metric("Total Storage (GB)", status["total_storage_gb"])
        col3.metric("Projected Deletions", status["projected_deletions"])
        col4.metric("Next Cleanup UTC", status["next_cleanup_utc"] or "Not scheduled")
    except Exception as exc:
        st.error(f"Retention status unavailable: {exc}")

    st.divider()
    st.subheader("Visual Analytics")

    defects_per_hour = query_defects_per_hour(sqlite_path=sqlite_path, lookback_hours=24)
    severity_df = query_severity_distribution(sqlite_path=sqlite_path)
    shift_df = query_defects_by_shift(sqlite_path=sqlite_path)
    machine_df = query_defects_by_machine(sqlite_path=sqlite_path)
    heatmap_df = query_hourly_heatmap(sqlite_path=sqlite_path, lookback_days=7)

    c1, c2 = st.columns(2)
    with c1:
        if not defects_per_hour.empty:
            fig = px.line(defects_per_hour, x="hour_bucket", y="defect_count", markers=True, title="Defects per Hour")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No hourly trend data.")

    with c2:
        if not severity_df.empty:
            fig = px.bar(severity_df, x="severity", y="count", title="Severity Distribution")
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No severity distribution data.")

    c3, c4 = st.columns(2)
    with c3:
        if not shift_df.empty:
            fig = px.bar(
                shift_df,
                x="shift",
                y="count",
                color="severity",
                barmode="group",
                title="Shift Comparison",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No shift data.")

    with c4:
        if not heatmap_df.empty:
            fig = px.density_heatmap(
                heatmap_df,
                x="hour_of_day",
                y="day_of_week",
                z="count",
                histfunc="sum",
                title="Hourly Heatmap (Defects vs Time)",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No heatmap data.")

    st.subheader("Defects by Machine")
    if not machine_df.empty:
        fig = px.bar(machine_df.head(10), x="machine", y="count")
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("Root Cause Insights")

    window_keys = ["last_hour", "last_shift", "last_24h"]
    available_windows = [k for k in window_keys if isinstance(rca.get(k), dict)]

    if not available_windows:
        st.info("RCA not available yet.")
    else:
        wcols = st.columns(len(available_windows))
        for col, key in zip(wcols, available_windows):
            wp = rca.get(key, {})
            conf = _window_confidence(wp)
            col.metric(
                f"{key.replace('_', ' ').title()} confidence",
                f"{conf:.2f}",
                delta=f"{int(wp.get('event_count', 0))} events",
            )

        primary_window = "last_hour" if "last_hour" in available_windows else available_windows[0]
        top_insights = _window_top_insights(rca.get(primary_window, {}), top_n=5)

        st.markdown(f"**Top insights ({primary_window.replace('_', ' ').title()})**")
        if top_insights:
            for idx, item in enumerate(top_insights, start=1):
                st.markdown(
                    f"{idx}. {item.get('insight', 'n/a')}  "
                    f"`confidence: {item.get('confidence', 0):.2f} ({item.get('confidence_tag', 'n/a')})`"
                )
        else:
            st.caption("No generated insights for selected window.")

        st.markdown("**Time window breakdown**")
        breakdown_rows = []
        for key in available_windows:
            wp = rca.get(key, {})
            breakdown_rows.append(
                {
                    "window": key,
                    "events": int(wp.get("event_count", 0) or 0),
                    "overall_confidence": round(_window_confidence(wp), 4),
                    "window_start": wp.get("window_start", "n/a"),
                    "window_end": wp.get("window_end", "n/a"),
                }
            )
        st.dataframe(breakdown_rows, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Live Operations")
    live_df = query_live_events(sqlite_path=sqlite_path, limit=50)
    show_events_table(live_df)

    if not live_df.empty:
        event_ids = live_df["event_id"].astype(str).tolist()
        selected_id = st.selectbox("Select event for evidence preview", options=event_ids)
        row = live_df[live_df["event_id"].astype(str) == selected_id].iloc[0]

        p1, p2 = st.columns(2)
        with p1:
            st.write("**Overlay**")
            overlay_path = str(row.get("overlay_path", "") or "")
            if overlay_path and os.path.exists(overlay_path):
                try:
                    st.image(overlay_path, width="stretch")
                except Exception:
                    st.warning("🏷️ Overlay image pruned by retention policy")
            else:
                st.warning("🏷️ Overlay image pruned or unavailable")

        with p2:
            st.write("**Crop / Full Image**")
            crop_path = str(row.get("crop_path", "") or "")
            full_path = str(row.get("full_image_path", "") or "")
            display_img = crop_path if (crop_path and os.path.exists(crop_path)) else (full_path if (full_path and os.path.exists(full_path)) else None)
            if display_img:
                try:
                    st.image(display_img, width="stretch")
                except Exception:
                    st.warning("🏷️ Image pruned by retention policy")
            else:
                st.warning("🏷️ Image pruned or unavailable")

        narrative = str(row.get("vlm_narrative", "") or "").strip()
        st.markdown("**🏷️ Reasoning AI**")
        if narrative:
            st.info(f"AI Root-Cause Narrative\n\n{narrative}")
        else:
            st.caption("AI Root-Cause Narrative not available for this event.")

    if auto_refresh:
        # Flaw Fix (Issue 5): Keep the timer strictly local to the iframe context.
        # When Streamlit reruns on user input, this iframe is unmounted, 
        # naturally clearing the countdown window without violating the browser's Same-Origin Policy.
        import streamlit.components.v1 as components
        components.html(
            """
            <div style="display: none; width: 0px; height: 0px; visibility: hidden;">
                <script>
                    window.setTimeout(function() {
                        if (window.parent) {
                            window.parent.postMessage({type: 'streamlit:rerun'}, '*');
                        }
                    }, 5000);
                </script>
            </div>
            """,
            height=0,
            width=0,
        )


if __name__ == "__main__":
    main()
