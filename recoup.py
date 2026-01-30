import streamlit as st
from datetime import date
from sqlalchemy import text

from db import run_scalar, run_df
from semantic import RECoup_START_DATE, BANK_REVENUE_DEFAULT, BANK_ASSIGNMENT_DEFAULT
from semantic import get_source_relation
from utils import show_df

def _is_blank_sql(col: str) -> str:
    return f"NULLIF(BTRIM({col}), '') IS NULL"

def _not_blank_sql(col: str) -> str:
    return f"NULLIF(BTRIM({col}), '') IS NOT NULL"

def compute_powerpivot_metrics(engine, where_sql: str, params: dict,
                              bank_revenue: str = BANK_REVENUE_DEFAULT,
                              bank_assignment: str = BANK_ASSIGNMENT_DEFAULT,
                              *, rel: str):
    # NOTE: logic unchanged vs current file, but centralized
    total_deposit = run_scalar(engine, f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from public.gl_register
        where {where_sql}
    """, params, rel=rel)

    pending_recoup_debit = run_scalar(engine, f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.gl_register
        where {where_sql}
          and bill_no ilike '%recoup%'
          and {_is_blank_sql('status')}
          and coalesce(account,'') <> coalesce(bank,'')
          and "date" >= :recoup_start
          and coalesce(bank,'') <> :bank_assignment
    """, {**params, "recoup_start": RECoup_START_DATE, "bank_assignment": bank_assignment}, rel=rel)

    completed_recoup = run_scalar(engine, f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.gl_register
        where {where_sql}
          and bill_no ilike '%recoup%'
          and {_not_blank_sql('status')}
          and coalesce(account,'') <> coalesce(bank,'')
    """, params, rel=rel)

    pending_recoup_minus_deposit = run_scalar(engine, f"""
        with p as (
          select
            coalesce(sum(coalesce(debit_payment,0)),0) as p_debit,
            coalesce(sum(coalesce(credit_deposit,0)),0) as p_credit
          from public.gl_register
          where {where_sql}
            and bill_no ilike '%recoup%'
            and {_is_blank_sql('status')}
            and coalesce(account,'') <> coalesce(bank,'')
            and "date" >= :recoup_start
        )
        select (p_debit - p_credit) from p
    """, {**params, "recoup_start": RECoup_START_DATE}, rel=rel)

    recoup_amount_revenue_bank = run_scalar(engine, f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from public.gl_register
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_revenue
    """, {**params, "bank_revenue": bank_revenue}, rel=rel)

    revenue_exp_not_recoup = run_scalar(engine, f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.gl_register
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_revenue
    """, {**params, "bank_revenue": bank_revenue}, rel=rel)

    exp_recoup_from_assignment = run_scalar(engine, f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.gl_register
        where {where_sql}
          and bill_no ilike '%recoup%'
          and bank = :bank_assignment
    """, {**params, "bank_assignment": bank_assignment}, rel=rel)

    total_expenses_revenue_dr = run_scalar(engine, f"""
        select coalesce(sum(coalesce(debit_payment,0)),0)
        from public.gl_register
        where {where_sql}
          and head_name = 'Expense'
          and bank = :bank_revenue
    """, {**params, "bank_revenue": bank_revenue}, rel=rel)

    total_expenses_revenue_cr = run_scalar(engine, f"""
        select coalesce(sum(coalesce(credit_deposit,0)),0)
        from public.gl_register
        where {where_sql}
          and head_name = 'Expense'
          and bank = :bank_revenue
    """, {**params, "bank_revenue": bank_revenue}, rel=rel)

    return {
        "Total Deposit": total_deposit,
        "Payments to be Recoup (Pending Debit)": pending_recoup_debit,
        "Completed Recoup (Debit)": completed_recoup,
        "Pending Recoup - Deposit": pending_recoup_minus_deposit,
        "Recoup Amount (Revenue Bank Credit)": recoup_amount_revenue_bank,
        "Revenue Bank Recoup Debit": revenue_exp_not_recoup,
        "Recoup Debit (Assignment Bank)": exp_recoup_from_assignment,
        "Total Expenses Revenue Dr": total_expenses_revenue_dr,
        "Total Expenses Revenue Cr": total_expenses_revenue_cr,
    }

def render_recoup_kpi_tab(engine, filters: dict, *, rel: str):
    st.subheader("Recoup KPIs (PowerPivot/DAX equivalent)")
    c0, c1 = st.columns(2)
    with c0:
        bank_revenue = st.text_input("Revenue Bank (for specific KPIs)", value=BANK_REVENUE_DEFAULT)
    with c1:
        bank_assignment = st.text_input("Assignment Bank (exclude / specific KPIs)", value=BANK_ASSIGNMENT_DEFAULT)

    where_sql = filters["where_sql"]
    params = filters["params"]

    kpis = compute_powerpivot_metrics(engine, where_sql, params,
                                     bank_revenue=bank_revenue,
                                     bank_assignment=bank_assignment,
                                     rel=rel)

    cols = st.columns(3)
    items = list(kpis.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 3]:
            st.metric(k, f"{v:,.0f}")

    st.divider()
    st.caption("Pending Recoup (Net) by Head")

    pending_by_head_sql = f"""
        select head_name,
               coalesce(sum(coalesce(debit_payment,0) - coalesce(credit_deposit,0)),0) as pending_net
        from public.gl_register
        where {where_sql}
          and bill_no ilike '%recoup%'
          and {_is_blank_sql('status')}
          and coalesce(account,'') <> coalesce(bank,'')
          and "date" >= :recoup_start
        group by 1
        order by 2 desc
        limit 100
    """
    df_pending = run_df(engine, pending_by_head_sql, {**params, "recoup_start": RECoup_START_DATE}, ["Head", "Pending Net"], rel=rel)
    show_df(df_pending, label_col="Head")
