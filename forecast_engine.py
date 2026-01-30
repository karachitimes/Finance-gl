
import streamlit as st
import pandas as pd
from db import run_df
from semantic import get_source_relation

def render_forecast_engine(engine, f, *, rel):
    st.subheader("Forecast Engine")
    st.caption("Revenue & Expense Prediction (Statistical Baseline Model)")

    rel0 = rel or get_source_relation(engine)

    # Revenue history
    sql_rev = f"""
        select date_trunc('month', date)::date as month,
               sum(credit_deposit) as revenue
        from {rel0}
        where func_code='Revenue'
        group by 1
        order by 1
    """

    # Expense history
    sql_exp = f"""
        select date_trunc('month', date)::date as month,
               sum(net_flow) as expense
        from {rel0}
        group by 1
        order by 1
    """

    df_rev = run_df(engine, sql_rev, f["params"], rel=rel0)
    df_exp = run_df(engine, sql_exp, f["params"], rel=rel0)

    if df_rev.empty or df_exp.empty:
        st.warning("Not enough historical data for forecasting.")
        return

    # Simple rolling forecast (baseline model)
    df_rev["forecast"] = df_rev["revenue"].rolling(3).mean()
    df_exp["forecast"] = df_exp["expense"].rolling(3).mean()

    st.markdown("### Revenue Forecast (Baseline)")
    st.dataframe(df_rev.tail(12))

    st.markdown("### Expense Forecast (Baseline)")
    st.dataframe(df_exp.tail(12))

    st.info("Model: Rolling Mean Forecast (Phase-2 baseline model). Can be upgraded to ARIMA / ML later.")
