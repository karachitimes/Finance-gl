
import streamlit as st
from db import run_df
from ai.anomaly_engine import detect_anomalies

def render_ai_dashboard(engine, f, rel):
    st.subheader("AI Intelligence Layer")

    sql = f'''
        select date, sum(net_flow) as expense
        from {rel}
        group by date
        order by date
    '''
    df = run_df(engine, sql, f["params"], rel=rel)

    if df.empty:
        st.warning("No data for AI analysis")
        return

    df = detect_anomalies(df, "expense")
    st.metric("Anomalies Detected", int(df["is_anomaly"].sum()))
    st.dataframe(df[df["is_anomaly"]==True])
