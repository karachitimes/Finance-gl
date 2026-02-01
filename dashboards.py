import streamlit as st
import pandas as pd

from db import run_df, run_scalar
from semantic import get_source_relation, pick_view, expense_amount_expr, cashflow_net_expr, cashflow_dir_expr, tb_amount_expr
from semantic import REL_REV, REL_EXP, REL_CF
from filters import build_where_from_ui
from utils import show_df

def render_revenue_tab(engine, f):
    st.subheader("Revenue (Monthly)")
    rel_sem = get_source_relation(engine)
    rel_rev = pick_view(engine, REL_REV, rel_sem)

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["head_no"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"],
        func_override="Revenue",
    )
    where_sql = " and ".join(where) if where else "1=1"

    extra = ""
    if rel_rev.endswith("v_finance_semantic"):
        extra = "and is_revenue = true"
    elif rel_rev.endswith("gl_register"):
        extra = "and func_code = 'Revenue' and coalesce(credit_deposit,0) > 0"

    sql = f"""
    select date_trunc('month', "date")::date as month,
           to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
           sum(coalesce(credit_deposit,0)) as revenue
    from {rel_rev}
    where {where_sql}
      {extra}
    group by 1,2
    order by 1
    """
    df_rev = run_df(engine, sql, params, ["Month","Month Label","Revenue"], rel=rel_sem)
    if df_rev.empty:
        st.info("No revenue rows found for selected filters/date range.")
        return
    show_df(df_rev, label_col="Month Label")
    st.line_chart(df_rev.set_index("Month")["Revenue"])
    st.success(f"Total Revenue: {df_rev['Revenue'].sum():,.0f} PKR")

def render_expense_tab(engine, f):
    st.subheader("Expenses (Monthly)")
    rel_sem = get_source_relation(engine)
    rel_exp = pick_view(engine, REL_EXP, rel_sem)

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"
    amt_expr = expense_amount_expr(engine, rel_exp)

    extra = ""
    if rel_exp.endswith("v_finance_semantic"):
        extra = "and is_expense = true"
    elif rel_exp.endswith("gl_register"):
        extra = "and lower(trim(coalesce(column1,''))) = 'expense'"

    sql = f"""
    select date_trunc('month', "date")::date as month,
           to_char(date_trunc('month', "date"), 'Mon-YY') as month_label,
           sum({amt_expr}) as expense_outflow
    from {rel_exp}
    where {where_sql}
      {extra}
    group by 1,2
    order by 1
    """
    df_exp = run_df(engine, sql, params, ["Month","Month Label","Expense Outflow"], rel=rel_sem)
    if df_exp.empty:
        st.info("No expense rows found for selected filters/date range.")
        return
    show_df(df_exp, label_col="Month Label")
    st.line_chart(df_exp.set_index("Month")["Expense Outflow"])
    st.success(f"Total Expense Outflow: {df_exp['Expense Outflow'].sum():,.0f} PKR")

def render_cashflow_tab(engine, f):
    st.subheader("Cashflow Summary (By Bank & Direction)")
    rel_sem = get_source_relation(engine)
    rel_cf = pick_view(engine, REL_CF, rel_sem)

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"

    if rel_cf.endswith("v_cashflow"):
        net_expr = "coalesce(net_flow,0)"
        dir_expr = "direction"
    else:
        net_expr = cashflow_net_expr(engine, rel_cf)
        dir_expr = cashflow_dir_expr(engine, rel_cf, net_expr)

    sql = f"""
    select coalesce(bank, 'UNKNOWN') as bank,
           {dir_expr} as direction,
           sum({net_expr}) as amount
    from {rel_cf}
    where {where_sql}
    group by 1,2
    order by 1,2
    """
    df_cf = run_df(engine, sql, params, ["Bank","Direction","Amount"], rel=rel_sem)
    if df_cf.empty:
        st.info("No rows found for selected filters/date range.")
        return
    show_df(df_cf, label_col="Bank")
    inflow = df_cf[df_cf["Direction"]=="in"]["Amount"].sum()
    outflow = df_cf[df_cf["Direction"]=="out"]["Amount"].sum()
    st.success(f"Inflow: {inflow:,.0f} PKR  |  Outflow: {abs(outflow):,.0f} PKR  |  Net: {(inflow+outflow):,.0f} PKR")

def render_trial_balance_tab(engine, f):
    st.subheader("Trial Balance (As of To Date)")
    rel_sem = get_source_relation(engine)

    where = ['"date" <= :dt']
    params = {"dt": f["dt"]}

    if f["bank"] != "ALL":
        where.append("bank = :bank"); params["bank"] = f["bank"]
    if f["head"] != "ALL":
        where.append("head_name = :head_name"); params["head_name"] = f["head"]
    if f["account"] != "ALL":
        where.append("account = :account"); params["account"] = f["account"]
    if f["func_code"] != "ALL":
        where.append("func_code = :func_code"); params["func_code"] = f["func_code"]

    amt_expr = tb_amount_expr(engine, rel_sem)
    sql = f"""
    select account,
           sum({amt_expr}) as balance
    from {rel_sem}
    where {' and '.join(where)}
    group by 1
    order by 1
    """
    df_tb = run_df(engine, sql, params, ["Account","Balance"], rel=rel_sem)
    if df_tb.empty:
        st.info("No rows found for trial balance with current filters.")
        return
    show_df(df_tb, label_col="Account")
    st.success(f"Net (sum of balances): {df_tb['Balance'].sum():,.0f} PKR")
