import streamlit as st
from db import run_df
from ai.anomaly_engine import detect_anomalies

def render_ai_dashboard(engine, f, rel):
    st.subheader("AI Intelligence Layer")
    st.caption("Anomaly detection uses IsolationForest if available; otherwise z-score fallback.")

    sql = f"""
        select "date"::date as date, coalesce(sum(net_flow),0) as expense
        from {rel}
        where "date" between %(df)s and %(dt)s
        group by 1
        order by 1
    """
    df = run_df(engine, sql, f["params"], rel=rel)

    if df is None or df.empty:
        st.warning("No data for AI analysis in current filters.")
        return

    contamination = st.slider("Anomaly sensitivity (contamination %)", 1, 10, 2) / 100.0
    z = st.slider("Z-score threshold (fallback)", 2.0, 6.0, 3.5, 0.1)

    df = detect_anomalies(df, "expense", contamination=contamination, z=z)

    st.metric("Anomalies Detected", int(df["is_anomaly"].sum()))
    st.write("Method:", df["anomaly_method"].iloc[0] if "anomaly_method" in df.columns and not df.empty else "unknown")

    st.dataframe(df[df["is_anomaly"]].sort_values("date", ascending=False))
    st.divider()
    st.dataframe(df.tail(90))
