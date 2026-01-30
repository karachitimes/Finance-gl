
import streamlit as st
from db import run_df, run_scalar
from semantic import get_source_relation
from utils import show_df

def render_revenue_intelligence(engine, f, *, rel):
    st.subheader("Revenue Intelligence")

    rel0 = rel or get_source_relation(engine)

    sql = f"""
        select head_name, sum(credit_deposit) as revenue
        from {rel0}
        where func_code='Revenue'
        group by head_name
        order by revenue desc
    """
    df = run_df(engine, sql, f["params"], rel=rel0)

    total = df["revenue"].sum() if not df.empty else 0
    top3 = df.head(3)["revenue"].sum() if not df.empty else 0

    st.metric("Revenue Concentration (Top 3 %)", round((top3/total)*100,2) if total else 0)
    show_df(df)
