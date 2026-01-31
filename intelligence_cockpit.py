import streamlit as st
import pandas as pd

from db import run_df
from ai.anomaly_engine import detect_anomalies
from ai.explain_panel import render_explain_panel


def render_intelligence_cockpit(engine, f, *, rel):
    st.subheader("🧠 Intelligence Cockpit")

    sql = f"""
        select date_trunc('month',"date")::date as month,
               coalesce(sum(coalesce(net_flow,0)),0) as expense
        from {rel}
        where {f.get("where_sql","1=1")}
        group by 1
        order by 1
    """

    df = run_df(engine, sql, f.get("params",{}), rel=rel)

    if df is None or df.empty:
        st.warning("No data available.")
        return

    df["expense"] = pd.to_numeric(df["expense"], errors="coerce").fillna(0)

    sensitivity = st.slider("Anomaly sensitivity", 2.5, 6.0, 3.5, 0.1)

    try:
        df = detect_anomalies(df, "expense", z=sensitivity)
    except TypeError:
        df = detect_anomalies(df, "expense")

    st.metric("Anomalous months", int(df["is_anomaly"].sum()))

    st.dataframe(df[df["is_anomaly"]])

    st.divider()

    # 🔍 WHY drilldown panel
    render_explain_panel(engine, f, rel=rel, df_anom=df)
