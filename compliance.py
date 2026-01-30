import streamlit as st
import pandas as pd

from db import run_df, run_scalar
from filters import build_where_from_ui
from semantic import (
    get_source_relation,
    pick_view,
    has_column,
    expense_amount_expr,
    REL_EXP,
    REL_REV,
)
from utils import show_df


def _blank_sql(col: str) -> str:
    return f"nullif(trim(coalesce({col},'')),'') is null"


def render_compliance_tab(engine, f, *, rel: str):
    st.subheader("Compliance Scanner")
    st.caption("Rule-based checks that surface likely posting/data issues. Uses the same filters/date range as the dashboards.")

    rel_sem = get_source_relation(engine)
    rel0 = rel or rel_sem

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"

    checks = [
        "Expense without Head",
        "Payments without Bill Reference",
        "Revenue without func_code = Revenue",
        "Recoup pending older than N days",
        "Duplicate Voucher/Reference",
        "Negative Expense Amount",
    ]
    check = st.selectbox("Select check", checks, index=0)

    # -----------------------------
    # Expense without Head
    # -----------------------------
    if check == "Expense without Head":
        rele = pick_view(engine, REL_EXP, rel0)
        exp_expr = expense_amount_expr(engine, rele)

        sql = f"""
            select "date", bank, account, pay_to,
                   head_name, voucher_no, reference_no,
                   {exp_expr} as amount,
                   description
            from {rele}
            where {where_sql}
              and ({_blank_sql('head_name')})
            order by "date" desc
            limit 500
        """
        df = run_df(engine, sql, params, rel=rele)
        show_df(df)
        return

    # -----------------------------
    # Payments without Bill Reference
    # -----------------------------
    if check == "Payments without Bill Reference":
        # Works on base relation (semantic or gl_register)
        sql = f"""
            select "date", bank, account, head_name, pay_to,
                   voucher_no, reference_no, bill_no, status,
                   coalesce(debit_payment,0) as debit_payment,
                   coalesce(credit_deposit,0) as credit_deposit,
                   description
            from {rel0}
            where {where_sql}
              and coalesce(debit_payment,0) > 0
              and ({_blank_sql('bill_no')})
            order by "date" desc
            limit 500
        """
        df = run_df(engine, sql, params, rel=rel0)
        show_df(df)
        return

    # -----------------------------
    # Revenue rows missing func_code = 'Revenue'
    # -----------------------------
    if check == "Revenue without func_code = Revenue":
        relr = pick_view(engine, REL_REV, rel0)
        # If func_code missing, fallback to credit_deposit sign only
        func_ok = "coalesce(func_code,'') <> 'Revenue'" if has_column(engine, relr, "func_code") else "1=1"
        sql = f"""
            select "date", bank, account, head_name, pay_to,
                   func_code,
                   coalesce(credit_deposit,0) as credit_deposit,
                   description
            from {relr}
            where {where_sql}
              and coalesce(credit_deposit,0) > 0
              and {func_ok}
            order by "date" desc
            limit 500
        """
        df = run_df(engine, sql, params, rel=relr)
        show_df(df)
        return

    # -----------------------------
    # Recoup pending older than N days
    # -----------------------------
    if check == "Recoup pending older than N days":
        days = st.number_input("Pending age threshold (days)", min_value=1, max_value=3650, value=30, step=5)
        sql = f"""
            select "date", bank, account, head_name, pay_to,
                   bill_no, status, voucher_no, reference_no,
                   (coalesce(debit_payment,0) - coalesce(credit_deposit,0)) as amount,
                   (current_date - "date") as age_days,
                   description
            from {rel0}
            where {where_sql}
              and bill_no = 'Recoup'
              and ({_blank_sql('status')})
              and (current_date - "date") >= :days
            order by "date" asc
            limit 500
        """
        params2 = dict(params)
        params2["days"] = int(days)
        df = run_df(engine, sql, params2, rel=rel0)
        show_df(df)
        if not df.empty and "amount" in df.columns:
            st.warning(f"Total pending amount older than {days} days: {pd.to_numeric(df['amount'], errors='coerce').fillna(0).sum():,.0f} PKR")
        return

    # -----------------------------
    # Duplicate voucher/reference
    # -----------------------------
    if check == "Duplicate Voucher/Reference":
        key = st.selectbox("Match key", ["voucher_no", "reference_no"], index=0)
        if not has_column(engine, rel0, key):
            st.info(f"Column {key} not available in {rel0}.")
            return
        sql = f"""
            select {key} as key,
                   count(*) as rows,
                   min("date") as first_date,
                   max("date") as last_date
            from {rel0}
            where {where_sql}
              and not ({_blank_sql(key)})
            group by 1
            having count(*) > 1
            order by rows desc, last_date desc
            limit 200
        """
        df = run_df(engine, sql, params, ["Key","Rows","First Date","Last Date"], rel=rel0)
        show_df(df, label_col="Key")
        return

    # -----------------------------
    # Negative expense amount
    # -----------------------------
    rele = pick_view(engine, REL_EXP, rel0)
    exp_expr = expense_amount_expr(engine, rele)
    sql = f"""
        select "date", bank, account, head_name, pay_to,
               {exp_expr} as amount,
               debit_payment, credit_deposit,
               description
        from {rele}
        where {where_sql}
          and ({exp_expr}) < 0
        order by "date" desc
        limit 500
    """
    df = run_df(engine, sql, params, rel=rele)
    show_df(df)
