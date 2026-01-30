
import streamlit as st
import pandas as pd
from db import run_df
from semantic import get_source_relation

def render_scenario_engine(engine, f, *, rel):
    st.subheader("Scenario Engine")
    st.caption("What-if Simulations")

    rel0 = rel or get_source_relation(engine)

    # Base values
    sql = f"""
        select
            sum(case when func_code='Revenue' then credit_deposit else 0 end) as revenue,
            sum(net_flow) as expense
        from {rel0}
    """

    df = run_df(engine, sql, f["params"], rel=rel0)

    if df.empty:
        st.warning("No data for scenario simulation.")
        return

    base_rev = float(df.iloc[0]["revenue"])
    base_exp = float(df.iloc[0]["expense"])

    st.markdown("### Scenario Inputs")
    rev_change = st.slider("Revenue Change (%)", -50, 50, 0)
    exp_change = st.slider("Expense Change (%)", -50, 50, 0)
    recoup_delay = st.slider("Recoup Delay Impact (%)", 0, 50, 0)

    sim_rev = base_rev * (1 + rev_change/100)
    sim_exp = base_exp * (1 + exp_change/100)
    sim_net = sim_rev - sim_exp

    st.markdown("### Scenario Output")
    st.metric("Simulated Revenue", round(sim_rev,2))
    st.metric("Simulated Expense", round(sim_exp,2))
    st.metric("Simulated Net Result", round(sim_net,2))

    st.success("Scenario simulation complete (deterministic what-if model).")
