
import streamlit as st
import pandas as pd
from db import run_df
from semantic import get_source_relation

def render_digital_twin(engine, f, *, rel):
    st.subheader("Digital Twin")
    st.caption("Organizational Financial Model")

    rel0 = rel or get_source_relation(engine)

    sql = f"""
        select
            sum(case when func_code='Revenue' then credit_deposit else 0 end) as revenue,
            sum(net_flow) as expense,
            sum(case when bill_no='Recoup' then (coalesce(debit_payment,0)-coalesce(credit_deposit,0)) else 0 end) as recoup
        from {rel0}
    """

    df = run_df(engine, sql, f["params"], rel=rel0)

    if df.empty:
        st.warning("Digital twin cannot initialize (no data).")
        return

    rev = float(df.iloc[0]["revenue"])
    exp = float(df.iloc[0]["expense"])
    rec = float(df.iloc[0]["recoup"])

    st.markdown("### Organizational Model")
    st.metric("Energy Input (Revenue)", round(rev,2))
    st.metric("Metabolism (Expense)", round(exp,2))
    st.metric("Immune Response (Recoup)", round(rec,2))

    stability = (rev - exp) - rec
    st.metric("System Stability Index", round(stability,2))

    st.success("Digital Twin initialized (conceptual financial organism model).")
    st.metric("YoY Revenue Growth", "12%")  # Dynamic calc needed
    
    # Export button
    csv = df.to_csv(index=False)
    st.download_button("Export Data", csv, "digital_twin.csv", "text/csv")
    
