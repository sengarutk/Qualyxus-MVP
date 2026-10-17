from __future__ import annotations
import pandas as pd
import streamlit as st


def kpi_row(
    defects_per_hour: int,
    avg_confidence: float,
    top_defect_type: str,
    active_cameras: int,
    uptime_text: str,
):
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Defects/hour (shift)", defects_per_hour)
    c2.metric("Avg confidence", f"{avg_confidence:.3f}")
    c3.metric("Top defect type", top_defect_type)
    c4.metric("Active cameras", active_cameras)
    c5.metric("System uptime", uptime_text)


def show_events_table(df: pd.DataFrame):
    st.dataframe(df, width="stretch", height=420)
