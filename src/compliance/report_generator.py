from __future__ import annotations

import sqlite3
import uuid
import socket
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any
import pandas as pd

from jinja2 import Environment, FileSystemLoader

from src.analytics.root_cause import RootCauseConfig, RootCauseIntelligence
from src.analytics.esg_engine import ESGEngine
from src.compliance.iso_mapper import ISOMapper
from src.operations.action_tracker import ActionTracker
from src.config.config import Config


def _generate_report_metadata(start_time: str, end_time: str) -> dict[str, Any]:
    """Generate enterprise-grade report metadata."""
    return {
        'report_id': str(uuid.uuid4()),
        'generated_at_utc': datetime.utcnow().isoformat(),
        'hostname': socket.gethostname(),
        'system_version': 'QUALYXUS v2.1.0',  # Could be loaded from config
        'data_window_start': start_time,
        'data_window_end': end_time,
    }


def generate_audit_report(
    start_time: str,
    end_time: str,
    output_pdf_path: str,
):
    """Generate production-grade audit report from live operational data only."""
    cfg = Config.load()
    events_db_path = cfg.get("logging", "sqlite_path")
    actions_db_path = str(Path(cfg.get("logging", "sqlite_path")).parent / "actions.db")
    defects_dir = Path(cfg.get("camera", "defect_dir", default="data/live/defects"))

    # Generate report metadata
    report_metadata = _generate_report_metadata(start_time, end_time)

    # Step 1: Load filtered events from events.db
    events_df, data_validation = _load_runtime_events(events_db_path, start_time, end_time)
    if events_df.empty:
        raise RuntimeError(f"No runtime events found between {start_time} and {end_time}")

    # Step 2: Run fresh RCA
    rca_payload = _run_fresh_rca(events_db_path)

    # Step 3: Load actions for window
    actions = _load_actions_for_window(actions_db_path, start_time, end_time)

    # Step 4: Compute KPIs
    kpis = _compute_operational_kpis(events_df, actions, start_time, end_time)

    # Step 5: Compute camera-level analysis
    camera_analysis = _compute_camera_analysis(events_df)

    # Step 6: Compute time distribution
    time_distribution = _compute_time_distribution(events_df)

    # Step 7: Compute action performance metrics
    action_performance = _compute_action_performance(actions)

    # Step 8: Compute RCA stability metric
    rca_stability = _compute_rca_stability(rca_payload)

    # Step 9: Compute process context insights
    process_context = _compute_process_context_insights(events_df)

    # Extract RCA insights for use in key takeaways
    rca_insights = _extract_rca_insights(rca_payload)
    advanced_sections = _extract_advanced_rca_sections(rca_payload)
    systemic_anomaly_analysis = _load_systemic_anomaly_analysis(cfg)

    # Step 10: Compute comparison trends
    comparison_trends = _compute_comparison_trends(
        events_db_path, actions_db_path, events_df, actions, start_time, end_time
    )

    # Step 11: Generate key takeaways
    key_takeaways = _generate_key_takeaways(rca_insights, camera_analysis, time_distribution, kpis)

    # Step 12: Pull top defect images
    evidence_images = _pull_top_defect_images(defects_dir, events_df)

    # Step 12b: ESG impact summary
    esg_metrics = ESGEngine(config=cfg).get_impact_metrics(events_df)
    cognitive_rca_insights = _extract_cognitive_rca_insights(events_df)

    # Step 13: Generate PDF
    _generate_pdf_report(
        report_metadata=report_metadata,
        data_validation=data_validation,
        start_time=start_time,
        end_time=end_time,
        events_df=events_df,
        rca_payload=rca_payload,
        actions=actions,
        kpis=kpis,
        camera_analysis=camera_analysis,
        time_distribution=time_distribution,
        action_performance=action_performance,
        rca_stability=rca_stability,
        comparison_trends=comparison_trends,
        process_context=process_context,
        key_takeaways=key_takeaways,
        evidence_images=evidence_images,
        esg_metrics=esg_metrics,
        rca_insights=rca_insights,
        advanced_sections=advanced_sections,
        cognitive_rca_insights=cognitive_rca_insights,
        systemic_anomaly_analysis=systemic_anomaly_analysis,
        output_pdf_path=output_pdf_path,
    )

    # Step 13: Export JSON
    _export_json_report(
        report_metadata=report_metadata,
        data_validation=data_validation,
        kpis=kpis,
        rca_insights=rca_insights,
        advanced_sections=advanced_sections,
        camera_analysis=camera_analysis,
        time_distribution=time_distribution,
        action_performance=action_performance,
        rca_stability=rca_stability,
        comparison_trends=comparison_trends,
        esg_metrics=esg_metrics,
        key_takeaways=key_takeaways,
        output_json_path=str(Path(output_pdf_path).parent / "audit_report.json"),
    )


def _load_runtime_events(db_path: str, start_time: str, end_time: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load events from SQLite with strict runtime filtering and data validation."""
    query = """
        SELECT * FROM defect_events
        WHERE julianday(timestamp) BETWEEN julianday(?) AND julianday(?)
        AND source NOT LIKE 'synthetic%'
        AND source NOT LIKE 'demo%'
        AND source NOT LIKE 'train%'
        ORDER BY timestamp DESC
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[start_time, end_time])

    # Data validation metrics
    total_events_analyzed = len(df)
    invalid_rows_skipped = 0  # Since we're filtering at query level

    # Check for missing metadata
    required_cols = ['event_id', 'timestamp', 'severity', 'defect_type', 'confidence']
    missing_metadata_count = 0
    for col in required_cols:
        if col in df.columns:
            missing_metadata_count += df[col].isnull().sum()
        else:
            missing_metadata_count += total_events_analyzed

    missing_metadata_percent = (missing_metadata_count / (total_events_analyzed * len(required_cols)) * 100) if total_events_analyzed > 0 else 0

    data_validation = {
        'total_events_analyzed': total_events_analyzed,
        'invalid_rows_skipped': invalid_rows_skipped,
        'missing_metadata_percent': round(missing_metadata_percent, 2),
    }

    return df, data_validation


def _prepare_traceability_records(events_df: pd.DataFrame, rca_payload: dict[str, Any]) -> list[dict[str, Any]]:
    classifications = rca_payload.get("all_time", {}).get("classification_6m", []) or []
    default_clause = None
    if classifications:
        default_clause = ISOMapper.clause_for_category(classifications[0].get("classification_6m"))

    records: list[dict[str, Any]] = []
    sample_df = events_df.head(10).copy()
    for row in sample_df.to_dict("records"):
        records.append(
            {
                "event_id": row.get("event_id"),
                "timestamp": row.get("timestamp"),
                "source": row.get("source"),
                "defect_type": row.get("defect_type"),
                "severity": row.get("severity"),
                "audit_hash": row.get("audit_hash"),
                "iso_clause": row.get("iso_clause") or default_clause,
            }
        )
    return records


def _extract_cognitive_rca_insights(events_df: pd.DataFrame) -> list[dict[str, Any]]:
    if events_df.empty or "vlm_narrative" not in events_df.columns:
        return []
    subset = events_df[events_df["vlm_narrative"].fillna("").astype(str).str.len() > 0].head(10).copy()
    return subset[["event_id", "defect_type", "severity", "vlm_narrative"]].to_dict("records")


def _load_systemic_anomaly_analysis(cfg: Config) -> dict[str, Any] | None:
    path = Path(cfg.get("paths", "data_dir", default="data")) / "runtime" / "systemic_insight.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _run_fresh_rca(events_db_path: str) -> dict[str, Any]:
    """Run RootCauseIntelligence fresh from events DB."""
    rca_config = RootCauseConfig(sqlite_path=events_db_path)
    rca_intelligence = RootCauseIntelligence(rca_config)
    return rca_intelligence.run()


def _load_actions_for_window(actions_db_path: str, start_time: str, end_time: str) -> list[dict[str, Any]]:
    """Load actions from ActionTracker for the time window."""
    tracker = ActionTracker(sqlite_path=actions_db_path)
    return tracker.list_actions(from_ts=start_time, to_ts=end_time, limit=100)


def _compute_operational_kpis(
    events_df: pd.DataFrame,
    actions: list[dict[str, Any]],
    start_time: str,
    end_time: str
) -> dict[str, Any]:
    """Compute production KPIs from runtime data."""
    total_defects = len(events_df)

    # Time window in hours
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    window_hours = (end_dt - start_dt).total_seconds() / 3600
    defects_per_hour = total_defects / window_hours if window_hours > 0 else 0

    # Severity distribution
    severity_counts = events_df['severity'].value_counts().to_dict()

    # MTTR (Mean Time To Resolution) for actions
    resolved_actions = [a for a in actions if a.get('status') == 'closed']
    if resolved_actions:
        mttr_hours = []
        for action in resolved_actions:
            issued = pd.to_datetime(action['timestamp'])
            resolved = pd.to_datetime(action.get('updated_at', action['timestamp']))
            mttr_hours.append((resolved - issued).total_seconds() / 3600)
        avg_mttr = sum(mttr_hours) / len(mttr_hours)
    else:
        avg_mttr = 0

    # Action closure rate
    total_actions = len(actions)
    closed_actions = len(resolved_actions)
    closure_rate = (closed_actions / total_actions * 100) if total_actions > 0 else 0

    return {
        'total_defects': total_defects,
        'defects_per_hour': round(defects_per_hour, 2),
        'severity_distribution': severity_counts,
        'avg_mttr_hours': round(avg_mttr, 2),
        'action_closure_rate_percent': round(closure_rate, 1),
        'total_actions': total_actions,
    }


def _pull_top_defect_images(defects_dir: Path, events_df: pd.DataFrame) -> list[dict[str, str]]:
    """Pull top defect images from data/live/defects and all camera subdirectories."""
    images = []
    top_events = events_df.nlargest(5, 'severity')  # Top 5 by severity

    for _, event in top_events.iterrows():
        event_id = str(event['event_id'])

        # Prefer the evidence paths recorded on the event itself (crop > overlay > snapshot).
        image_path = None
        for col in ("crop_path", "overlay_path", "full_image_path"):
            candidate = event.get(col) if hasattr(event, "get") else None
            if isinstance(candidate, str) and candidate and Path(candidate).is_file():
                image_path = Path(candidate)
                break

        # Fall back to searching the defect directory (camera sub-folders included) for any
        # snapshot whose file name carries the event id, e.g. cam_0_defect_<event_id>_frame_0001.png
        if image_path is None and defects_dir.exists():
            for pattern in (f"{event_id}.*", f"*{event_id}*"):
                match = next((p for p in defects_dir.rglob(pattern) if p.is_file()), None)
                if match is not None:
                    image_path = match
                    break

        if image_path:
            images.append({
                'path': str(image_path.resolve()),
                'event_id': event_id,
                'severity': int(event['severity']),
                'defect_type': event['defect_type'],
            })

    return images


def _generate_pdf_report(
    report_metadata: dict[str, Any],
    data_validation: dict[str, Any],
    start_time: str,
    end_time: str,
    events_df: pd.DataFrame,
    rca_payload: dict[str, Any],
    actions: list[dict[str, Any]],
    kpis: dict[str, Any],
    camera_analysis: dict[str, Any],
    time_distribution: dict[str, Any],
    action_performance: dict[str, Any],
    rca_stability: dict[str, Any],
    comparison_trends: dict[str, Any],
    process_context: dict[str, Any],
    key_takeaways: list[str],
    evidence_images: list[dict[str, str]],
    esg_metrics: dict[str, Any],
    rca_insights: dict[str, Any],
    advanced_sections: dict[str, Any],
    cognitive_rca_insights: list[dict[str, Any]],
    systemic_anomaly_analysis: dict[str, Any] | None,
    output_pdf_path: str,
):
    """Generate structured PDF report."""
    from weasyprint import HTML

    template_dir = Path("src/compliance/templates")
    cfg = Config.load()
    template_dir = Path(cfg.get("paths", "data_dir", default="data")).resolve().parents[0] / "src" / "compliance" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report_template.html")
    traceability_records = _prepare_traceability_records(events_df, rca_payload)

    # Prepare data for template
    html = template.render(
        report_metadata=report_metadata,
        data_validation=data_validation,
        generated_at=report_metadata['generated_at_utc'],
        report_window_start=start_time,
        report_window_end=end_time,
        kpis=kpis,
        rca_insights=rca_insights,
        advanced_sections=advanced_sections,
        actions_summary=_summarize_actions(actions),
        camera_analysis=camera_analysis,
        time_distribution=time_distribution,
        action_performance=action_performance,
        rca_stability=rca_stability,
        comparison_trends=comparison_trends,
        process_context=process_context,
        key_takeaways=key_takeaways,
        evidence_images=evidence_images,
        esg_metrics=esg_metrics,
        cognitive_rca_insights=cognitive_rca_insights,
        systemic_anomaly_analysis=systemic_anomaly_analysis,
        events_sample=events_df.head(10).to_dict('records'),  # Sample for details
        traceability_records=traceability_records,
    )

    out_pdf = Path(output_pdf_path)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html).write_pdf(str(out_pdf))

    print(f"[QUALYXUS] ✅ Enterprise audit PDF generated at: {out_pdf.resolve()}")


def _export_json_report(
    report_metadata: dict[str, Any],
    data_validation: dict[str, Any],
    kpis: dict[str, Any],
    rca_insights: dict[str, Any],
    advanced_sections: dict[str, Any],
    camera_analysis: dict[str, Any],
    time_distribution: dict[str, Any],
    action_performance: dict[str, Any],
    rca_stability: dict[str, Any],
    comparison_trends: dict[str, Any],
    esg_metrics: dict[str, Any],
    key_takeaways: list[str],
    output_json_path: str,
):
    """Export full structured report to JSON."""

    report_data = {
        'metadata': report_metadata,
        'data_validation': data_validation,
        'kpis': kpis,
        'rca_insights': rca_insights,
        'advanced_sections': advanced_sections,
        'systemic_anomaly_analysis': _load_systemic_anomaly_analysis(Config.load()),
        'camera_analysis': camera_analysis,
        'time_distribution': time_distribution,
        'action_performance': action_performance,
        'rca_stability': rca_stability,
        'comparison_trends': comparison_trends,
        'esg_metrics': esg_metrics,
        'key_takeaways': key_takeaways,
    }

    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)

    print(f"[QUALYXUS] ✅ Enterprise audit JSON exported at: {out_json.resolve()}")


def _extract_rca_insights(rca_payload: dict[str, Any]) -> dict[str, Any]:
    """Extract key RCA insights for report."""
    all_time = rca_payload.get('all_time', {})
    correlations = all_time.get('correlations', {})
    anomalies = all_time.get('anomalies', {})

    top_machine = None
    if correlations.get('machine_id'):
        top_machine = correlations['machine_id'][0]

    top_defect = None
    if correlations.get('defect_type'):
        top_defect = correlations['defect_type'][0]

    spikes = anomalies.get('spikes', [])
    spike_count = len(spikes)

    return {
        'confidence': all_time.get('confidence', {}).get('overall_confidence', 0),
        'top_machine': top_machine,
        'top_defect_type': top_defect,
        'spike_count': spike_count,
        'classification_6m': all_time.get('classification_6m', []),
        'top_causal_factors': (all_time.get('causal_scoring', {}) or {}).get('ranked_factors', [])[:3],
    }


def _extract_advanced_rca_sections(rca_payload: dict[str, Any]) -> dict[str, Any]:
    all_time = rca_payload.get('all_time', {})
    temporal = (all_time.get('temporal_rca', {}) or {}).get('insights', [])
    control_limits = (all_time.get('control_limits', {}) or {}).get('metrics', {})
    predictive = all_time.get('predictive_maintenance', {}) or {}

    control_rows = []
    for signal, payload in control_limits.items():
        control_rows.append(
            {
                'signal': signal,
                'outside_control_pct': round(float(payload.get('defect_rate_outside_control_limits', 0.0) or 0.0) * 100, 2),
                'outside_warning_pct': round(float(payload.get('defect_rate_outside_warning_limits', 0.0) or 0.0) * 100, 2),
                'trend_violations': int(payload.get('trend_violations', 0) or 0),
                'safe_range': payload.get('safe_range', {}),
                'warning_limits': payload.get('warning_limits', {}),
                'control_limits': payload.get('control_limits', {}),
            }
        )

    return {
        'temporal_rca_insights': temporal,
        'control_limit_violations': control_rows,
        'predictive_risk_summary': predictive,
    }


def _summarize_actions(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actions for report."""
    status_counts = {}
    for action in actions:
        status = action.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1

    return {
        'total': len(actions),
        'by_status': status_counts,
        'recent_actions': actions[:5],  # Top 5 recent
    }


def _compute_camera_analysis(events_df: pd.DataFrame) -> dict[str, Any]:
    """Compute camera-level defect analysis."""
    if events_df.empty:
        return {'defects_by_camera': {}, 'severity_by_camera': {}}

    # Defects by camera
    defects_by_camera = events_df['source'].value_counts().to_dict()

    # Severity distribution by camera
    severity_by_camera = {}
    for camera in events_df['source'].unique():
        camera_events = events_df[events_df['source'] == camera]
        severity_by_camera[camera] = camera_events['severity'].value_counts().to_dict()

    return {
        'defects_by_camera': defects_by_camera,
        'severity_by_camera': severity_by_camera,
    }


def _compute_time_distribution(events_df: pd.DataFrame) -> dict[str, Any]:
    """Compute time-based defect distribution."""
    if events_df.empty:
        return {'defects_by_hour': {}, 'defects_by_shift': {}}

    # Ensure timestamp is datetime
    events_df = events_df.copy()
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

    # Defects by hour of day
    defects_by_hour = events_df['timestamp'].dt.hour.value_counts().sort_index().to_dict()

    # Defects by shift (assuming 8-hour shifts: 6-14, 14-22, 22-6)
    def get_shift(hour):
        if 6 <= hour < 14:
            return 'Morning (6-14)'
        elif 14 <= hour < 22:
            return 'Afternoon (14-22)'
        else:
            return 'Night (22-6)'

    events_df['shift'] = events_df['timestamp'].dt.hour.apply(get_shift)
    defects_by_shift = events_df['shift'].value_counts().to_dict()

    return {
        'defects_by_hour': defects_by_hour,
        'defects_by_shift': defects_by_shift,
    }


def _compute_action_performance(actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute action performance metrics."""
    if not actions:
        return {
            'avg_ack_time_minutes': 0,
            'avg_resolution_time_minutes': 0,
            'sla_compliance_percent': 0,
        }

    ack_times = []
    resolution_times = []
    sla_compliant = 0

    for action in actions:
        created = pd.to_datetime(action['timestamp'])
        updated = pd.to_datetime(action.get('updated_at', action['timestamp']))

        # Assume acknowledged if status changed from 'open'
        if action.get('status') != 'open':
            ack_time = (updated - created).total_seconds() / 60
            ack_times.append(ack_time)

        # Resolution time for closed actions
        if action.get('status') == 'closed':
            resolution_time = (updated - created).total_seconds() / 60
            resolution_times.append(resolution_time)

            # SLA compliance: assume 4 hours for resolution
            if resolution_time <= 240:  # 4 hours in minutes
                sla_compliant += 1

    avg_ack_time = sum(ack_times) / len(ack_times) if ack_times else 0
    avg_resolution_time = sum(resolution_times) / len(resolution_times) if resolution_times else 0
    sla_compliance = (sla_compliant / len(resolution_times) * 100) if resolution_times else 0

    return {
        'avg_ack_time_minutes': round(avg_ack_time, 1),
        'avg_resolution_time_minutes': round(avg_resolution_time, 1),
        'sla_compliance_percent': round(sla_compliance, 1),
    }


def _compute_rca_stability(rca_payload: dict[str, Any]) -> dict[str, Any]:
    """Compute RCA stability metrics."""
    all_time = rca_payload.get('all_time', {})
    confidence = all_time.get('confidence', {})
    
    overall_confidence = confidence.get('overall_confidence', 0)
    factor_confidence = confidence.get('factor_confidence', {})
    
    # Stability indicators
    stability_score = overall_confidence * 100  # Convert to percentage
    confidence_trend = "Stable"  # Could be computed from historical data
    
    # Top contributing factors
    top_factors = []
    if factor_confidence:
        sorted_factors = sorted(factor_confidence.items(), key=lambda x: x[1], reverse=True)
        top_factors = [factor for factor, conf in sorted_factors[:3]]
    
    return {
        'consistency_score': round(stability_score, 1),
        'confidence_trend': confidence_trend,
        'top_factors': top_factors,
        'overall_confidence': round(overall_confidence, 3),
    }


def _compute_comparison_trends(
    events_db_path: str,
    actions_db_path: str,
    events_df: pd.DataFrame,
    actions: list[dict[str, Any]],
    start_time: str,
    end_time: str
) -> dict[str, Any]:
    """Compute comparison metrics vs previous period."""
    # Calculate previous period (same duration before current window)
    current_start = pd.to_datetime(start_time)
    current_end = pd.to_datetime(end_time)
    duration = current_end - current_start

    prev_end = current_start
    prev_start = prev_end - duration

    # Get previous period data
    prev_events_df, _ = _load_runtime_events(
        events_db_path,
        prev_start.isoformat(),
        prev_end.isoformat()
    )

    prev_actions = _load_actions_for_window(
        actions_db_path,
        prev_start.isoformat(),
        prev_end.isoformat()
    )

    # Current period metrics
    current_defects = len(events_df)
    current_mttr = _calculate_mttr(actions)
    current_closure_rate = _calculate_closure_rate(actions)

    # Previous period metrics
    prev_defects = len(prev_events_df)
    prev_mttr = _calculate_mttr(prev_actions)
    prev_closure_rate = _calculate_closure_rate(prev_actions)

    # Calculate changes
    defects_change = _calculate_percent_change(current_defects, prev_defects)
    mttr_change = _calculate_percent_change(current_mttr, prev_mttr)
    closure_rate_change = _calculate_percent_change(current_closure_rate, prev_closure_rate)

    return {
        'defects_change': round(defects_change, 1),
        'mttr_change': round(mttr_change, 1),
        'closure_rate_change': round(closure_rate_change, 1),
    }


def _calculate_mttr(actions: list[dict[str, Any]]) -> float:
    """Calculate Mean Time To Resolution."""
    resolved_actions = [a for a in actions if a.get('status') == 'closed']
    if not resolved_actions:
        return 0

    mttr_hours = []
    for action in resolved_actions:
        issued = pd.to_datetime(action['timestamp'])
        resolved = pd.to_datetime(action.get('updated_at', action['timestamp']))
        mttr_hours.append((resolved - issued).total_seconds() / 3600)

    return sum(mttr_hours) / len(mttr_hours) if mttr_hours else 0


def _calculate_closure_rate(actions: list[dict[str, Any]]) -> float:
    """Calculate action closure rate."""
    if not actions:
        return 0
    closed_actions = len([a for a in actions if a.get('status') == 'closed'])
    return (closed_actions / len(actions)) * 100


def _calculate_percent_change(current: float, previous: float) -> float:
    """Calculate percentage change."""
    if previous == 0:
        return 0 if current == 0 else 100
    return ((current - previous) / previous) * 100


def _compute_process_context_insights(events_df: pd.DataFrame) -> dict[str, Any]:
    """Compute process context insights from telemetry data."""
    insights = {}

    # Average temperature during defects
    if 'temperature' in events_df.columns and not events_df['temperature'].isna().all():
        avg_temp = events_df['temperature'].mean()
        insights['average_temperature'] = round(avg_temp, 1) if pd.notna(avg_temp) else None

    # Vibration correlation
    if 'vibration' in events_df.columns and not events_df['vibration'].isna().all():
        avg_vib = events_df['vibration'].mean()
        insights['average_vibration'] = round(avg_vib, 3) if pd.notna(avg_vib) else None

    # Speed correlation
    if 'speed' in events_df.columns and not events_df['speed'].isna().all():
        avg_speed = events_df['speed'].mean()
        insights['average_speed'] = round(avg_speed, 1) if pd.notna(avg_speed) else None

    # Most affected machine
    if 'machine_id' in events_df.columns:
        machine_counts = events_df['machine_id'].value_counts()
        if not machine_counts.empty:
            top_machine = machine_counts.index[0]
            insights['most_affected_machine'] = top_machine

    return insights


def _generate_key_takeaways(
    rca_insights: dict[str, Any],
    camera_analysis: dict[str, Any],
    time_distribution: dict[str, Any],
    kpis: dict[str, Any]
) -> list[str]:
    """Generate 3 key takeaways."""
    takeaways = []

    # Top RCA driver
    if rca_insights.get('top_machine') and rca_insights['top_machine']:
        machine = rca_insights['top_machine'].get('machine_id', 'Unknown')
        lift = rca_insights['top_machine'].get('lift_ratio', 0)
        takeaways.append(f"Top RCA driver: Machine {machine} shows {lift:.2f}x higher defect rate than average")
    elif rca_insights.get('top_defect_type') and rca_insights['top_defect_type']:
        defect_type = rca_insights['top_defect_type'].get('defect_type', 'Unknown')
        lift = rca_insights['top_defect_type'].get('lift_ratio', 0)
        takeaways.append(f"Top RCA driver: {defect_type} defects occur {lift:.2f}x more frequently than average")
    else:
        takeaways.append("Top RCA driver: Analysis indicates uniform defect distribution across factors")

    # Most affected camera/shift
    defects_by_camera = camera_analysis.get('defects_by_camera', {})
    defects_by_shift = time_distribution.get('defects_by_shift', {})

    if defects_by_camera:
        top_camera = max(defects_by_camera.items(), key=lambda x: x[1])
        total_defects = kpis.get('total_defects', 1)
        percentage = (top_camera[1] / total_defects) * 100
        takeaways.append(f"Most affected camera: {top_camera[0]} accounts for {percentage:.1f}% of all defects")
    elif defects_by_shift:
        top_shift = max(defects_by_shift.items(), key=lambda x: x[1])
        total_defects = kpis.get('total_defects', 1)
        percentage = (top_shift[1] / total_defects) * 100
        takeaways.append(f"Most affected shift: {top_shift[0]} accounts for {percentage:.1f}% of all defects")
    else:
        takeaways.append("Most affected area: Defect distribution is balanced across cameras and shifts")

    # Biggest improvement opportunity
    closure_rate = kpis.get('action_closure_rate_percent', 0)
    mttr = kpis.get('avg_mttr_hours', 0)
    defect_rate = kpis.get('defects_per_hour', 0)

    if closure_rate < 80:
        takeaways.append(f"Biggest improvement opportunity: Action closure rate of {closure_rate}% indicates need for better follow-through")
    elif mttr > 2:
        takeaways.append(f"Biggest improvement opportunity: {mttr:.1f} hour average MTTR suggests faster response needed")
    elif defect_rate > 1:
        takeaways.append(f"Biggest improvement opportunity: Reduce defect rate from {defect_rate:.1f} defects/hour through targeted interventions")
    else:
        takeaways.append(f"Biggest improvement opportunity: Maintain current excellent performance with {defect_rate:.1f} defects/hour")

    return takeaways


if __name__ == "__main__":
    # Example usage - last 24 hours
    end_time = datetime.utcnow().isoformat()
    start_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()

    cfg = Config.load()
    reports_dir = Path(cfg.get("paths", "reports_dir"))
    output_path = str(reports_dir / "production_audit_report.pdf")

    generate_audit_report(
        start_time=start_time,
        end_time=end_time,
        output_pdf_path=output_path,
    )
