import streamlit as st
from db import run_df
from semantic import get_source_relation
from utils import show_df


def render_expense_intelligence(engine, f, *, rel):
    st.subheader("Expense Intelligence")

    rel0 = rel or get_source_relation(engine)

    # Use global filters built in app.py
    where_sql = f.get("where_sql", "1=1")
    params = f.get("params", {})

    # 1) Summary: vendor totals (filtered)
    sql = f"""
        select pay_to, sum(coalesce(net_flow,0)) as expense
        from {rel0}
        where {where_sql}
        group by pay_to
        order by expense desc
    """
    df = run_df(engine, sql, params, rel=rel0)

    if df is None or df.empty:
        st.info("No expense rows found for selected filters/date range.")
        return

    # KPIs
    total = float(df["expense"].sum())
    top5 = float(df.head(5)["expense"].sum())
    st.metric("Vendor Concentration Risk (Top 5 %)", round((top5 / total) * 100, 2) if total else 0)

    show_df(df)

    st.divider()

    # 2) Drill-down selector
    selected_vendor = st.selectbox(
        "Drill into vendor",
        options=df["pay_to"].astype(str).tolist(),
        index=0
    )

    # 3) Drill detail: underlying ledger rows for that vendor
    st.markdown(f"### 🔎 Drill-down: {selected_vendor}")

    sql_drill = f"""
        select
            "date"::date as date,
            bank,
            head_name,
            head_no,
            account,
            bill_no,
            ref,
            folio_chq_no,
            description,
            pay_to,
            coalesce(debit_payment,0) as debit_payment,
            coalesce(credit_deposit,0) as credit_deposit,
            coalesce(net_flow,0) as net_flow
        from {rel0}
        where {where_sql}
          and coalesce(pay_to,'') = :pay_to
        order by "date" desc
        limit 5000
    """
    drill_params = dict(params)
    drill_params["pay_to"] = selected_vendor

    df_drill = run_df(engine, sql_drill, drill_params, rel=rel0)
    show_df(df_drill)
