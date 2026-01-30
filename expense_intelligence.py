
import streamlit as st
from db import run_df
from semantic import get_source_relation
from utils import show_df

def render_expense_intelligence(engine, f, *, rel):
    st.subheader("Expense Intelligence")

    rel0 = rel or get_source_relation(engine)

    sql = f"""
        select pay_to, sum(net_flow) as expense
        from {rel0}
        group by pay_to
        order by expense desc
    """
    df = run_df(engine, sql, f["params"], rel=rel0)

    total = df["expense"].sum() if not df.empty else 0
    top5 = df.head(5)["expense"].sum() if not df.empty else 0

    st.metric("Vendor Concentration Risk (Top 5 %)", round((top5/total)*100,2) if total else 0)
    show_df(df)
