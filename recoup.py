import streamlit as st
import pandas as pd
import numpy as np

from db import run_df
from filters import build_where_from_ui
from semantic import get_source_relation
from utils import show_df


def _recoup_amount_expr(engine, rel: str) -> str:
    # Recoup amounts are safely represented as debit - credit in the base table/view
    return "(coalesce(debit_payment,0) - coalesce(credit_deposit,0))"


def render_recoup_kpi_tab(engine, f, *, rel: str):
    """Backward-compatible entry point used by app.py.

    This upgraded version includes:
    - pending/completed summary
    - aging buckets
    - cycle time (approx)
    - recovery efficiency
    - loss risk (aged pending)
    """
    render_recoup_intelligence_tab(engine, f, rel=rel)


def render_recoup_intelligence_tab(engine, f, *, rel: str):
    st.subheader("Recoup Intelligence Panel")
    st.caption("Recoup = bill_no='Recoup'. status blank = pending; status filled = completed.")

    rel_sem = get_source_relation(engine)
    rel0 = rel or rel_sem

    where, params, _ = build_where_from_ui(
        f["df"], f["dt"], f["bank"], f["head"], f["account"], f["attribute"], f["func_code"],
        fy_label=f["fy_label"], func_override=None
    )
    where_sql = " and ".join(where) if where else "1=1"
    amt_expr = _recoup_amount_expr(engine, rel0)

    # ---- 1) Pending vs Completed summary
    sql_summary = f"""
        select
          case when nullif(trim(coalesce(status,'')),'') is null then 'pending' else 'completed' end as recoup_state,
          coalesce(sum({amt_expr}),0) as amount
        from {rel0}
        where {where_sql}
          and bill_no = 'Recoup'
        group by 1
        order by 1
    """
    df_sum = run_df(engine, sql_summary, params, ["State","Amount"], rel=rel0)
    if df_sum.empty:
        st.info("No recoup rows for selected filters/date range.")
        return
    show_df(df_sum, label_col="State")
    pending_amt = float(df_sum[df_sum["State"]=="pending"]["Amount"].sum())
    completed_amt = float(df_sum[df_sum["State"]=="completed"]["Amount"].sum())
    total_amt = pending_amt + completed_amt
    recovery_rate = (completed_amt / total_amt) if total_amt else 0.0
    st.success(f"Recovery rate (amount): {recovery_rate:.1%}  |  Pending: {pending_amt:,.0f} PKR  |  Completed: {completed_amt:,.0f} PKR")

    st.divider()

    # ---- 2) Aging buckets (pending)
    st.subheader("Aging buckets (Pending)")
    buckets = [
        (0, 30, "0–30"),
        (31, 60, "31–60"),
        (61, 90, "61–90"),
        (91, 180, "91–180"),
        (181, 99999, "180+"),
    ]

    # Compute age in SQL and bucket in python for portability
    sql_pending = f"""
        select
          "date"::date as date,
          (current_date - "date")::int as age_days,
          bank, account, head_name, pay_to, voucher_no, reference_no,
          {amt_expr} as amount,
          description
        from {rel0}
        where {where_sql}
          and bill_no = 'Recoup'
          and nullif(trim(coalesce(status,'')),'') is null
        order by "date" asc
        limit 5000
    """
    df_pending = run_df(engine, sql_pending, params, rel=rel0)
    if df_pending.empty:
        st.success("No pending recoup rows in selected range.")
    else:
        df_pending["amount"] = pd.to_numeric(df_pending["amount"], errors="coerce").fillna(0.0)
        df_pending["age_days"] = pd.to_numeric(df_pending["age_days"], errors="coerce").fillna(0).astype(int)

        def _bucket(age: int) -> str:
            for lo, hi, label in buckets:
                if lo <= age <= hi:
                    return label
            return "180+"

        df_pending["bucket"] = df_pending["age_days"].apply(_bucket)
        df_b = df_pending.groupby("bucket", as_index=False)["amount"].sum().rename(columns={"amount":"Amount"})
        # keep bucket order
        order = [b[2] for b in buckets]
        df_b["bucket"] = pd.Categorical(df_b["bucket"], categories=order, ordered=True)
        df_b = df_b.sort_values("bucket")
        df_b = df_b.rename(columns={"bucket":"Aging Bucket"})
        show_df(df_b, label_col="Aging Bucket")

        # Loss risk = pending older than threshold
        st.subheader("Loss risk (aged pending)")
        risk_days = st.slider("Flag pending older than (days)", min_value=30, max_value=720, value=180, step=30)
        risk_df = df_pending[df_pending["age_days"] >= int(risk_days)].copy()
        risk_amt = float(risk_df["amount"].sum()) if not risk_df.empty else 0.0
        st.warning(f"Pending older than {risk_days} days: {risk_amt:,.0f} PKR")
        if not risk_df.empty:
            st.caption("Top at-risk payees (by amount)")
            top = (risk_df.groupby(risk_df.get("pay_to", "pay_to"), as_index=False)["amount"].sum()
                   .sort_values("amount", ascending=False)
                   .head(25))
            top.columns = ["Payee","Amount"]
            show_df(top, label_col="Payee")

    st.divider()

    # ---- 3) Cycle time (approx) + Completion efficiency
    st.subheader("Cycle time (approx)")
    st.caption("Approximation: group by reference_no (fallback voucher_no). Start = earliest date, Complete = earliest date where status is filled.")

    key_col = "reference_no" if "reference_no" in (df_pending.columns if df_pending is not None else []) else "reference_no"
    # Build from SQL so it works even if df_pending is empty
    sql_cycle = f"""
        with r as (
            select
              coalesce(nullif(trim(coalesce(reference_no,'')),''), nullif(trim(coalesce(voucher_no,'')),''), 'UNKNOWN') as rec_key,
              "date"::date as dt,
              nullif(trim(coalesce(status,'')),'') as status_norm,
              {amt_expr} as amount
            from {rel0}
            where {where_sql}
              and bill_no = 'Recoup'
        ),
        agg as (
            select
              rec_key,
              min(dt) as start_date,
              min(case when status_norm is not null then dt end) as completed_date,
              sum(case when status_norm is null then amount else 0 end) as pending_amount,
              sum(case when status_norm is not null then amount else 0 end) as completed_amount
            from r
            group by 1
        )
        select
          rec_key,
          start_date,
          completed_date,
          (case when completed_date is null then null else (completed_date - start_date) end) as cycle_days,
          pending_amount,
          completed_amount
        from agg
        order by start_date desc
        limit 1000
    """
    df_cycle = run_df(engine, sql_cycle, params, rel=rel0)
    if df_cycle.empty:
        st.info("Not enough recoup rows to compute cycle time.")
        return

    # Summary stats
    df_cycle["cycle_days"] = pd.to_numeric(df_cycle["cycle_days"], errors="coerce")
    completed = df_cycle[df_cycle["cycle_days"].notna()].copy()
    if completed.empty:
        st.info("No completed recoup keys detected in selected range.")
    else:
        p50 = float(completed["cycle_days"].median())
        p90 = float(completed["cycle_days"].quantile(0.9))
        avg = float(completed["cycle_days"].mean())
        st.success(f"Cycle time (days): median {p50:.0f} | p90 {p90:.0f} | average {avg:.0f}")

    show = df_cycle.copy()
    show = show.rename(columns={
        "rec_key": "Recoup Key",
        "start_date": "Start Date",
        "completed_date": "Completed Date",
        "cycle_days": "Cycle Days",
        "pending_amount": "Pending Amount",
        "completed_amount": "Completed Amount",
    })
    show_df(show, label_col="Recoup Key")
